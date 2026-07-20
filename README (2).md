# Dobot Vision & Control

A Python + Tkinter desktop app for controlling a Dobot robotic arm alongside
a live computer-vision pipeline. Provides jog controls for the arm, and a
switchable set of camera-based detection modes (object, colour, ArUco
markers).

## Features

- **Live camera feed** with a resizable, scrollable window layout
- **Object detection** — contour-based placeholder, easy to swap for a real
  model (e.g. YOLO)
- **Colour detection** — HSV-based tracking of a colour you pick from a
  colour wheel
- **ArUco marker detection** — real-time detection with ID + optional pose
  estimation (`cv2.aruco`)
- **Arm jog controls** — X / Y / Z movement, Home, and gripper open/close
- **Simulation mode** — runs and is fully clickable even with no Dobot
  connected, so you can test the UI without hardware

## Requirements

- Python 3.9+
- A webcam
- (Optional) a Dobot Magician / M1 connected over USB/serial, for real
  arm movement

## Installation

```bash
git clone <your-repo-url>
cd dobot-vision-control
pip install -r requirements.txt
```

> **Important:** `requirements.txt` installs `opencv-contrib-python`, which
> includes the `cv2.aruco` module needed for marker detection. Do **not**
> also install plain `opencv-python` alongside it — the two conflict. If you
> already have `opencv-python` installed (e.g. as a dependency of
> `mediapipe`), remove it first:
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

## Usage

```bash
python dobot_control_ui.py
```

The app opens maximized. If no Dobot is connected, it automatically runs in
**simulation mode** — every arm command is printed to the console instead of
sent to hardware, so the UI is fully testable without a physical arm.

### Connecting a real Dobot

Open `dobot_control_ui.py` and set your serial port near the top of the
file:

```python
DOBOT_PORT = "COM3"          # Windows example
# DOBOT_PORT = "/dev/ttyUSB0"  # Linux example
```

### Detection modes

Click any of the three mode buttons in the **Detection Modes** panel. Modes
are mutually exclusive — starting one automatically stops any other that's
running.

| Mode | Status | Notes |
|---|---|---|
| Object Detection | Working (placeholder) | Contour/edge-based; swap in a real model when ready |
| Colour Detection | Working | Pick a target colour with the colour-wheel button first |
| ArUco Detection | Working | Uses `DICT_4X4_50` by default; generate a test marker with `generate_marker_image()` in `aruco_detection.py` |

### Generating a test ArUco marker

```python
from aruco_detection import generate_marker_image
generate_marker_image(0)   # saves aruco_marker_0.png
```
Print it (or display it on a phone screen) and hold it up to the camera
with ArUco Detection running.

## Project structure

```
.
├── dobot_control_ui.py     # Main Tkinter application
├── aruco_detection.py      # ArUco marker detection module
├── requirements.txt
├── .gitignore
└── README.md
```

## Configuration reference

All at the top of `dobot_control_ui.py`:

| Variable | Purpose | Default |
|---|---|---|
| `DOBOT_PORT` | Serial port for the arm; `None` = simulation mode | `None` |
| `CAMERA_INDEX` | Which webcam to use | `0` |
| `JOG_STEP_MM` | Distance moved per jog button click | `5` |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | Camera feed resolution | `640x480` |

In `aruco_detection.py`:

| Variable | Purpose | Default |
|---|---|---|
| `ARUCO_DICT_NAME` | Which ArUco dictionary to detect | `DICT_4X4_50` |
| `MARKER_LENGTH_MM` | Physical marker size, for pose estimation | `40.0` |
| `CAMERA_MATRIX`, `DIST_COEFFS` | Camera calibration, for real-world pose/distance | `None` (pose estimation skipped) |

## Roadmap / known placeholders

- [ ] Replace contour-based object detection with a trained model
- [ ] Camera calibration workflow for ArUcgio pose estimation
- [ ] Wire detected object/marker positions into actual arm movement
      (`ArmController.move_to_pixel_target()` exists but isn't auto-triggered)

