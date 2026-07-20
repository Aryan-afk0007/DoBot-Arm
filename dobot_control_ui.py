"""
Dobot Vision & Control UI

A polished Tkinter front-end for a Dobot robotic arm that provides:
  - A live camera feed in a themed card
  - Start/Stop OBJECT DETECTION (contour-based placeholder)
  - Start/Stop COLOUR DETECTION (HSV-based, pick a target colour)
  - Start/Stop ArUco MARKER DETECTION (cv2.aruco, with optional pose estimation)
  - Jog controls for X / Y / Z movement, Home, and Gripper open/close
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, colorchooser
from tkinter import font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageTk

from aruco_detection import detect_aruco

# =========================================================
# CONFIG
# =========================================================
DOBOT_PORT = None          # e.g. "COM3" or "/dev/ttyUSB0" — None = simulation mode
CAMERA_INDEX = 0            # webcam index
JOG_STEP_MM = 5              # distance moved per arrow-button click
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# =========================================================
# COLOUR THEME — "Aurora" dark theme
# =========================================================
COL_BG = "#0b0e17"            # app background
COL_PANEL = "#131826"         # card background
COL_PANEL_ALT = "#1a2136"     # secondary / inset background
COL_BORDER = "#232b42"        # hairline borders
COL_TEXT = "#eef1fb"
COL_MUTED = "#8891ab"

COL_ACCENT = "#00e5c7"        # aurora teal (primary)
COL_ACCENT_DARK = "#00b9a1"
COL_ACCENT2 = "#ff4fa3"       # aurora pink (secondary / gradient partner)

COL_GREEN = "#35d488"
COL_GREEN_DARK = "#23a868"
COL_RED = "#ff5d6c"
COL_RED_DARK = "#e0404f"
COL_BLUE = "#4da3ff"
COL_BLUE_DARK = "#2f7fd6"
COL_ORANGE = "#ffb454"
COL_ORANGE_DARK = "#d9932f"
COL_TEAL = "#22d3c8"
COL_TEAL_DARK = "#17a89f"
COL_DISABLED = "#3a4054"

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"


# =========================================================
# COLOUR / GRADIENT HELPERS
# =========================================================
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c))) for c in rgb)


def _lerp_color(c1, c2, t):
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def _round_points(w, h, r):
    r = min(r, w / 2, h / 2)
    return [
        r, 0, w - r, 0, w, 0, w, r,
        w, h - r, w, h, w - r, h, r, h,
        0, h, 0, h - r, 0, r, 0, 0,
    ]


# =========================================================
# DOBOT ARM WRAPPER
# =========================================================
class ArmController:
    """Thin wrapper around pydobot. Falls back to a simulated arm
    (prints actions instead of moving) if no port is configured or
    the connection fails, so the UI is always usable for testing."""

    def __init__(self, port=None):
        self.simulated = True
        self.device = None
        self.pose = {"x": 200.0, "y": 0.0, "z": 50.0}
        self.gripper_open = True

        if port:
            try:
                import pydobot
                # >>> PLUG IN HERE: real connection
                self.device = pydobot.Dobot(port=port)
                self.simulated = False
            except Exception as exc:
                print(f"[ArmController] Could not connect to Dobot on {port}: {exc}")
                print("[ArmController] Falling back to simulation mode.")

    def move_relative(self, dx=0, dy=0, dz=0):
        self.pose["x"] += dx
        self.pose["y"] += dy
        self.pose["z"] += dz

        if self.simulated:
            print(f"[SIM] move_relative dx={dx} dy={dy} dz={dz} -> {self.pose}")
        else:
            # >>> PLUG IN HERE: real relative move
            x, y, z = self.pose["x"], self.pose["y"], self.pose["z"]
            self.device.move_to(x, y, z, 0, wait=False)

    def home(self):
        self.pose = {"x": 200.0, "y": 0.0, "z": 50.0}
        if self.simulated:
            print("[SIM] home()")
        else:
            # >>> PLUG IN HERE: real home command
            self.device.home()

    def move_to_pixel_target(self, cx, cy, frame_w, frame_h):
     
        offset_x = (cx - frame_w / 2) / frame_w * JOG_STEP_MM * 2
        offset_y = (cy - frame_h / 2) / frame_h * JOG_STEP_MM * 2
        self.move_relative(dx=offset_y, dy=-offset_x)

    def toggle_gripper(self):
        self.gripper_open = not self.gripper_open
        state = "open" if self.gripper_open else "closed"
        if self.simulated:
            print(f"[SIM] gripper -> {state}")
        else:
            # >>> PLUG IN HERE: real gripper command
            self.device.grip(self.gripper_open)
        return state

    def close(self):
        if not self.simulated and self.device:
            self.device.close()


# =========================================================
# VISION HELPERS
# =========================================================
def detect_objects(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    annotated = frame.copy()
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        detections.append({"bbox": (x, y, w, h), "center": (cx, cy)})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (46, 204, 113), 2)
        cv2.circle(annotated, (cx, cy), 4, (255, 93, 93), -1)

    return annotated, detections


def detect_color(frame, target_bgr, tolerance=25):
    """HSV-based colour detection: masks pixels close to target_bgr and
    returns bounding info for the largest matching blob."""
    target = np.uint8([[target_bgr]])
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0][0]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([max(0, int(target_hsv[0]) - tolerance), 80, 80])
    upper = np.array([min(179, int(target_hsv[0]) + tolerance), 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = frame.copy()
    detection = None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
            detection = {"bbox": (x, y, w, h), "center": (cx, cy)}
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (77, 163, 255), 2)
            cv2.circle(annotated, (cx, cy), 4, (255, 93, 93), -1)

    return annotated, detection


# =========================================================
# STYLE HELPERS — rounded, canvas-drawn widgets
# =========================================================
class RoundedButton(tk.Canvas):
    """A canvas-drawn pill/rounded button. Supports the same
    .config(text=..., bg=..., state=...) calls a plain tk.Button
    would, so it can be dropped in without touching call sites."""

    def __init__(self, parent, text, command, base, dark,
                 width=180, height=44, radius=16, fg="#0c0f16",
                 font=(FONT_FAMILY, 10, "bold"), state="normal"):
        parent_bg = parent["bg"] if "bg" in parent.keys() else COL_PANEL
        super().__init__(parent, width=width, height=height,
                          bg=parent_bg, highlightthickness=0, bd=0)
        self.command = command
        self.base = base
        self.dark = dark
        self.fg = fg
        self.radius = radius
        self.font = font
        self.w = width
        self.h = height
        self._state = state
        self._text = text

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self._draw(self.base)
        self.configure(cursor="hand2" if state == "normal" else "arrow")

    def _draw(self, fill_color):
        self.delete("all")
        pts = _round_points(self.w, self.h, self.radius)
        self.create_polygon(pts, smooth=True, fill=fill_color, outline=fill_color)
        text_color = self.fg if self._state == "normal" else "#7d8399"
        self.create_text(self.w / 2, self.h / 2, text=self._text,
                          fill=text_color, font=self.font)

    def _on_enter(self, _e):
        if self._state == "normal":
            self._draw(self.dark)

    def _on_leave(self, _e):
        if self._state == "normal":
            self._draw(self.base)

    def _on_click(self, _e):
        if self._state == "normal" and self.command:
            self.command()

    def config(self, **kwargs):
        text = kwargs.pop("text", None)
        bg = kwargs.pop("bg", None)
        state = kwargs.pop("state", None)
        if bg is not None:
            self.base = bg
        if text is not None:
            self._text = text
        if state is not None:
            self._state = state
            self.configure(cursor="hand2" if state == "normal" else "arrow")
        self._draw(self.base)
        if kwargs:
            super().config(**kwargs)

    configure = config


def make_button(parent, text, command, base, dark, width=180, height=44, state="normal"):
    return RoundedButton(parent, text, command, base, dark,
                          width=width, height=height, state=state)


def card(parent, bg=COL_PANEL, accent=None, **kwargs):
    """A card panel with a hairline border and an optional coloured
    accent strip along the top edge for a bit of visual identity."""
    outer = tk.Frame(parent, bg=bg, highlightbackground=COL_BORDER,
                      highlightthickness=1, **kwargs)
    if accent:
        tk.Frame(outer, bg=accent, height=3).pack(fill="x", side="top")
    return outer


def section_title(parent, text, bg=COL_PANEL, accent=COL_ACCENT):
    wrap = tk.Frame(parent, bg=bg)
    tk.Frame(wrap, bg=accent, width=4, height=18).pack(side="left", padx=(0, 8))
    tk.Label(wrap, text=text, bg=bg, fg=COL_TEXT,
              font=(FONT_FAMILY, 12, "bold")).pack(side="left")
    return wrap


# =========================================================
# MAIN APPLICATION
# =========================================================
class DobotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dobot Vision & Control")
        self.root.configure(bg=COL_BG)
        self.root.resizable(True, True)
        try:
            self.root.state("zoomed")          # Windows / most Linux window managers
        except tk.TclError:
            self.root.attributes("-zoomed", True)   # macOS fallback

        self.arm = ArmController(DOBOT_PORT)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.running = True
        self.mode = None  # None | "object" | "color" | "aruco"
        self.target_color_bgr = (0, 0, 255)  # default: red

        self._build_layout()

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------
    # UI LAYOUT
    # ---------------------------------------------------
    def _build_layout(self):
        # ---- Header bar (gradient, canvas-drawn) ----
        header = tk.Canvas(self.root, height=64, highlightthickness=0, bg=COL_BG)
        header.pack(fill="x", side="top")
        mode_note = "SIMULATION MODE" if self.arm.simulated else f"CONNECTED · {DOBOT_PORT}"

        def render_header(_event=None):
            header.delete("all")
            w = header.winfo_width()
            h = header.winfo_height()
            if w < 2:
                return
            steps = 48
            for i in range(steps):
                t = i / steps
                color = _lerp_color(COL_ACCENT2, COL_ACCENT, t)
                x0 = int(w * i / steps)
                x1 = int(w * (i + 1) / steps) + 1
                header.create_rectangle(x0, 0, x1, h, fill=color, outline=color)

            header.create_text(24, h / 2, anchor="w",
                                text="\u25c9  Dobot Vision & Control",
                                fill="#08121a", font=(FONT_FAMILY, 17, "bold"))

            badge_text = mode_note
            badge_font = (FONT_FAMILY, 9, "bold")
            text_w = badge_font_width(badge_text)
            pad = 16
            bw, bh = text_w + pad * 2, 26
            bx1 = w - bw - 20
            by1 = (h - bh) / 2
            pts = _round_points(bw, bh, bh / 2)
            pts = [pts[i] + (bx1 if i % 2 == 0 else by1) for i in range(len(pts))]
            header.create_polygon(pts, smooth=True, fill="#08121a", outline="#08121a")
            header.create_text(bx1 + bw / 2, by1 + bh / 2, text=badge_text,
                                fill=COL_ACCENT, font=badge_font)

        def badge_font_width(text):
            f = tkfont.Font(family=FONT_FAMILY, size=9, weight="bold")
            return f.measure(text)

        header.bind("<Configure>", render_header)

        # -----------------------------------------------
        # SCROLLABLE CONTAINER
        # -----------------------------------------------
        scroll_container = tk.Frame(self.root, bg=COL_BG)
        scroll_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(scroll_container, bg=COL_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_container, orient="vertical", command=self.canvas.yview,
                                  bg=COL_PANEL_ALT, troughcolor=COL_BG,
                                  activebackground=COL_ACCENT, highlightthickness=0, bd=0)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        body = tk.Frame(self.canvas, bg=COL_BG)
        body_window = self.canvas.create_window((0, 0), window=body, anchor="nw")

        def _update_scrollregion(_event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _resize_body_to_canvas(event):
            # keep the inner frame as wide as the visible canvas so the
            # left/right cards can still stretch horizontally
            self.canvas.itemconfig(body_window, width=event.width)

        body.bind("<Configure>", _update_scrollregion)
        self.canvas.bind("<Configure>", _resize_body_to_canvas)

        def _on_mousewheel(event):
            # Windows/Mac send event.delta; Linux sends Button-4/5 instead
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
            else:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Windows / Mac
        self.canvas.bind_all("<Button-4>", _on_mousewheel)      # Linux scroll up
        self.canvas.bind_all("<Button-5>", _on_mousewheel)      # Linux scroll down

        content = tk.Frame(body, bg=COL_BG)
        content.pack(fill="both", expand=True, padx=18, pady=18)

        # =========================================================
        # LEFT CARD: camera + detection controls
        # =========================================================
        left = card(content, accent=COL_ACCENT)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))

        section_title(left, "Live Camera", accent=COL_ACCENT).pack(
            anchor="w", padx=18, pady=(14, 10))

        video_wrap = tk.Frame(left, bg="#000000", highlightbackground=COL_ACCENT,
                               highlightthickness=2)
        video_wrap.pack(padx=18)
        self.video_label = tk.Label(video_wrap, bg="black")
        self.video_label.pack()

        self.status_var = tk.StringVar(value="Mode: Idle")
        status_wrap = tk.Frame(left, bg=COL_PANEL_ALT, highlightbackground=COL_BORDER,
                                highlightthickness=1)
        status_wrap.pack(anchor="w", padx=18, pady=(14, 6), fill="x")
        self.status_label = tk.Label(status_wrap, textvariable=self.status_var, bg=COL_PANEL_ALT,
                                      fg=COL_ACCENT, font=(FONT_FAMILY, 10, "bold"),
                                      anchor="w", padx=12, pady=8)
        self.status_label.pack(fill="x")

        tk.Frame(left, bg=COL_BORDER, height=1).pack(fill="x", padx=18, pady=(14, 14))

        section_title(left, "Detection Modes", accent=COL_TEAL).pack(
            anchor="w", padx=18, pady=(0, 10))

        detect_row = tk.Frame(left, bg=COL_PANEL)
        detect_row.pack(padx=18, fill="x")

        self.obj_btn = make_button(detect_row, "\u25b6  Object Detection", self.toggle_object_detection,
                                    COL_GREEN, COL_GREEN_DARK, width=220, height=46)
        self.obj_btn.grid(row=0, column=0, padx=(0, 10), pady=6, sticky="w")

        self.color_btn = make_button(detect_row, "\u25b6  Colour Detection", self.toggle_color_detection,
                                      COL_BLUE, COL_BLUE_DARK, width=220, height=46)
        self.color_btn.grid(row=0, column=1, pady=6, sticky="w")

        self.aruco_btn = make_button(detect_row, "\u25b6  ArUco Detection", self.toggle_aruco_detection,
                                      COL_TEAL, COL_TEAL_DARK, width=220, height=46)
        self.aruco_btn.grid(row=1, column=0, pady=(0, 10), sticky="w")

        pick_row = tk.Frame(left, bg=COL_PANEL)
        pick_row.pack(padx=18, pady=(4, 18), fill="x")
        pick_btn = make_button(pick_row, "\U0001F3A8  Pick Target Colour", self.pick_color,
                                COL_PANEL_ALT, "#262f4a", width=220, height=42)
        pick_btn.pack(side="left")
        self.color_swatch = tk.Canvas(pick_row, width=30, height=30, bg="red",
                                       highlightbackground=COL_TEXT, highlightthickness=1)
        self.color_swatch.pack(side="left", padx=12)

        # =========================================================
        # RIGHT CARD: arm controls
        # =========================================================
        right = card(content, accent=COL_ACCENT2, width=360)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        section_title(right, "Arm Movement", accent=COL_ACCENT2).pack(
            anchor="w", padx=18, pady=(14, 12))

        jog_card = card(right, bg=COL_PANEL_ALT)
        jog_card.pack(padx=18, pady=(0, 16), fill="x")

        jog = tk.Frame(jog_card, bg=COL_PANEL_ALT)
        jog.pack(pady=16)

        def jog_btn(parent, text, cmd):
            return make_button(parent, text, cmd, COL_BLUE, COL_BLUE_DARK, width=68, height=48)

        jog_btn(jog, "Y+", lambda: self.jog(dy=JOG_STEP_MM)).grid(row=0, column=1, pady=4)
        jog_btn(jog, "X-", lambda: self.jog(dx=-JOG_STEP_MM)).grid(row=1, column=0, padx=4)
        make_button(jog, "\u2302", self.go_home, COL_ACCENT2, "#c93d84",
                    width=68, height=48).grid(row=1, column=1, padx=4)
        jog_btn(jog, "X+", lambda: self.jog(dx=JOG_STEP_MM)).grid(row=1, column=2, padx=4)
        jog_btn(jog, "Y-", lambda: self.jog(dy=-JOG_STEP_MM)).grid(row=2, column=1, pady=4)

        z_row = tk.Frame(jog_card, bg=COL_PANEL_ALT)
        z_row.pack(pady=(0, 16))
        jog_btn(z_row, "Z+", lambda: self.jog(dz=JOG_STEP_MM)).pack(side="left", padx=8)
        jog_btn(z_row, "Z-", lambda: self.jog(dz=-JOG_STEP_MM)).pack(side="left", padx=8)

        self.gripper_btn = make_button(right, "\u270b  Gripper: Open", self.toggle_gripper,
                                        COL_ORANGE, COL_ORANGE_DARK, width=270, height=46)
        self.gripper_btn.pack(padx=18, pady=(0, 18))

        tk.Frame(right, bg=COL_BORDER, height=1).pack(fill="x", padx=18, pady=(0, 16))

        section_title(right, "Position", accent=COL_TEAL).pack(
            anchor="w", padx=18, pady=(0, 8))

        pose_card = card(right, bg=COL_PANEL_ALT)
        pose_card.pack(padx=18, fill="x")
        self.pose_var = tk.StringVar(value=self._pose_text())
        tk.Label(pose_card, textvariable=self.pose_var, bg=COL_PANEL_ALT, fg=COL_ACCENT,
                 font=(FONT_MONO, 11), justify="left", padx=14, pady=12).pack(anchor="w")

    def _pose_text(self):
        p = self.arm.pose
        return f"X: {p['x']:6.1f}\nY: {p['y']:6.1f}\nZ: {p['z']:6.1f}"

    # ---------------------------------------------------
    # DETECTION BUTTON HANDLERS
    # ---------------------------------------------------
    def _reset_detection_buttons(self):
        self.obj_btn.config(text="\u25b6  Object Detection", bg=COL_GREEN)
        self.color_btn.config(text="\u25b6  Colour Detection", bg=COL_BLUE)
        self.aruco_btn.config(text="\u25b6  ArUco Detection", bg=COL_TEAL)

    def toggle_object_detection(self):
        if self.mode == "object":
            self.mode = None
            self.status_var.set("Mode: Idle")
            self._reset_detection_buttons()
        else:
            self._reset_detection_buttons()
            self.mode = "object"
            self.obj_btn.config(text="\u25a0  Stop Object Detection", bg=COL_RED)
            self.status_var.set("Mode: Object Detection")

    def toggle_color_detection(self):
        if self.mode == "color":
            self.mode = None
            self.status_var.set("Mode: Idle")
            self._reset_detection_buttons()
        else:
            self._reset_detection_buttons()
            self.mode = "color"
            self.color_btn.config(text="\u25a0  Stop Colour Detection", bg=COL_RED)
            self.status_var.set("Mode: Colour Detection")

    def toggle_aruco_detection(self):
        if self.mode == "aruco":
            self.mode = None
            self.status_var.set("Mode: Idle")
            self._reset_detection_buttons()
        else:
            self._reset_detection_buttons()
            self.mode = "aruco"
            self.aruco_btn.config(text="\u25a0  Stop ArUco Detection", bg=COL_RED)
            self.status_var.set("Mode: ArUco Detection")

    def pick_color(self):
        rgb, _ = colorchooser.askcolor(title="Pick target colour")
        if rgb:
            r, g, b = [int(c) for c in rgb]
            self.target_color_bgr = (b, g, r)
            self.color_swatch.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    # ---------------------------------------------------
    # ARM BUTTON HANDLERS
    # ---------------------------------------------------
    def jog(self, dx=0, dy=0, dz=0):
        self.arm.move_relative(dx, dy, dz)
        self.pose_var.set(self._pose_text())

    def go_home(self):
        self.arm.home()
        self.pose_var.set(self._pose_text())

    def toggle_gripper(self):
        state = self.arm.toggle_gripper()
        icon = "\u270b" if state == "open" else "\u270a"
        self.gripper_btn.config(text=f"{icon}  Gripper: {state.capitalize()}")

    # ---------------------------------------------------
    # VIDEO LOOP (runs in background thread)
    # ---------------------------------------------------
    def _video_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            detection_center = None

            if self.mode == "object":
                frame, detections = detect_objects(frame)
                if detections:
                    detection_center = detections[0]["center"]
                    # >>> PLUG IN HERE: decide when/whether to move the arm
                    # e.g. self.arm.move_to_pixel_target(*detection_center, FRAME_WIDTH, FRAME_HEIGHT)

            elif self.mode == "color":
                frame, detection = detect_color(frame, self.target_color_bgr)
                if detection:
                    detection_center = detection["center"]
                    # >>> PLUG IN HERE: decide when/whether to move the arm
                    # e.g. self.arm.move_to_pixel_target(*detection_center, FRAME_WIDTH, FRAME_HEIGHT)

            elif self.mode == "aruco":
                frame, detections = detect_aruco(frame)
                if detections:
                    ids_found = ", ".join(str(d["id"]) for d in detections)
                    self.status_var.set(f"Mode: ArUco Detection — found ID(s) {ids_found}")
                    detection_center = detections[0]["center"]
                    # >>> PLUG IN HERE: decide when/whether to move the arm
                    # e.g. self.arm.move_to_pixel_target(*detection_center, FRAME_WIDTH, FRAME_HEIGHT)
                else:
                    self.status_var.set("Mode: ArUco Detection — no marker in view")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            time.sleep(0.03)

    # ---------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------
    def _on_close(self):
        self.running = False
        time.sleep(0.1)
        if self.cap.isOpened():
            self.cap.release()
        self.arm.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DobotUI(root)
    root.mainloop()