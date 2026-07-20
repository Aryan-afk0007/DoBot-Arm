"""
ArUco Marker Detection
=======================

Drop this into your Dobot Vision & Control app as a fourth detection
mode, alongside detect_objects() / detect_color() / detect_qr().

Requires:
    pip install opencv-contrib-python
    (opencv-contrib-python is required for the cv2.aruco module —
     plain opencv-python does NOT include it. Uninstall opencv-python
     first if you have it, to avoid a conflicting install.)


"""

import cv2
import numpy as np

ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50

MARKER_LENGTH_MM = 40.0

CAMERA_MATRIX = None    # e.g. np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
DIST_COEFFS = None      # e.g. np.array([k1, k2, p1, p2, k3])


# =========================================================
# DETECTOR SETUP (handles both old and new OpenCV APIs)
# =========================================================
def _build_detector(dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return ("new", detector)
    else:
        # OpenCV < 4.7
        params = cv2.aruco.DetectorParameters_create()
        return ("old", (aruco_dict, params))


_API_VERSION, _DETECTOR = _build_detector(ARUCO_DICT_NAME)


# =========================================================
# MAIN DETECTION FUNCTION
# =========================================================
def detect_aruco(frame, marker_length_mm=MARKER_LENGTH_MM,
                  camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS):
    """
    Detect ArUco markers in a frame.

    Returns (annotated_frame, detections) where each detection is:
        {
            "id": int,
            "corners": (4, 2) array of pixel corner points,
            "center": (cx, cy) pixel center,
            "distance_mm": float or None,   # only if camera calibrated
            "rvec": rotation vector or None,
            "tvec": translation vector or None,
        }

    This matches the same (frame, detections) shape as
    detect_objects() / detect_color() / detect_qr(), so it can be
    dropped straight into the existing video loop as a new mode.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _API_VERSION == "new":
        corners, ids, _rejected = _DETECTOR.detectMarkers(gray)
    else:
        aruco_dict, params = _DETECTOR
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    annotated = frame.copy()
    detections = []

    if ids is None:
        return annotated, detections

    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    do_pose = marker_length_mm is not None and camera_matrix is not None and dist_coeffs is not None
    rvecs = tvecs = None
    if do_pose:
        rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length_mm, camera_matrix, dist_coeffs
        )

    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i].reshape(4, 2)
        cx, cy = pts.mean(axis=0)

        detection = {
            "id": int(marker_id),
            "corners": pts,
            "center": (float(cx), float(cy)),
            "distance_mm": None,
            "rvec": None,
            "tvec": None,
        }

        if do_pose:
            rvec, tvec = rvecs[i][0], tvecs[i][0]
            detection["rvec"] = rvec
            detection["tvec"] = tvec
            detection["distance_mm"] = float(np.linalg.norm(tvec))

            # Draw a small 3D axis on the marker to visualize orientation
            cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec,
                               marker_length_mm * 0.5)

        # Label with ID (and distance, if known) above the marker
        label = f"ID {detection['id']}"
        if detection["distance_mm"] is not None:
            label += f"  {detection['distance_mm']:.0f}mm"
        top_left = tuple(pts[0].astype(int))
        cv2.putText(annotated, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 84), 2)

        detections.append(detection)

    return annotated, detections


# =========================================================
def generate_marker_image(marker_id, size_px=400, dict_name=ARUCO_DICT_NAME):
    """Save a printable marker PNG, e.g. generate_marker_image(7)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)

    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    else:
        img = cv2.aruco.drawMarker(aruco_dict, marker_id, size_px)

    filename = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(filename, img)
    return filename
