#Used media pipe
"""Human pose estimation and measurement assistance via MediaPipe Pose.

Limitations (documented honestly in the UI):
- A single ordinary photograph cannot yield accurate physical measurements.
  Camera distance, angle, lens, clothing and pose all distort pixel scale.
- Estimates below are only returned when a full body is visible AND the user
  provides their real height, which fixes the pixel->centimetre scale.
"""
import math
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    _POSE = mp.solutions.pose
    _DRAW = mp.solutions.drawing_utils
    _AVAILABLE = True
except Exception:  # pragma: no cover
    _AVAILABLE = False

# MediaPipe landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def analyze_person(rgb: np.ndarray, height_cm: Optional[float] = None) -> dict:
    if not _AVAILABLE:
        return {"available": False, "detected": False, "error": "MediaPipe is not installed."}

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())

    out = {
        "available": True,
        "detected": False,
        "image_width": int(w),
        "image_height": int(h),
        "quality": {
            "blur_variance": round(blur, 1),
            "brightness": round(bright, 1),
            "resolution_ok": bool(min(w, h) >= 360),
            "sharp": bool(blur >= 60),
            "well_lit": bool(30 <= bright <= 235),
        },
        "landmarks": {},
        "estimates": None,
        "overlay_b64": None,
        "warnings": [],
    }

    with _POSE.Pose(static_image_mode=True, model_complexity=1,
                    min_detection_confidence=0.5) as pose:
        res = pose.process(rgb)

    if res.pose_landmarks is None:
        out["warnings"].append("No human pose detected. Use a clear, front-facing, full-body or upper-body photo.")
        return out

    lm = res.pose_landmarks.landmark
    pts = {}
    for i in (NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
              L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        pts[i] = (lm[i].x * w, lm[i].y * h, lm[i].visibility)

    def dist(a, b):
        return math.dist((pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]))

    shoulder_px = dist(L_SHOULDER, R_SHOULDER)
    hip_px = dist(L_HIP, R_HIP)
    torso_px = abs(((pts[L_HIP][1] + pts[R_HIP][1]) / 2) - ((pts[L_SHOULDER][1] + pts[R_SHOULDER][1]) / 2))
    full_body = all(pts[i][2] >= 0.5 for i in (L_ANKLE, R_ANKLE))
    # Rough head-to-ankle span in px for scale
    nose_y = pts[NOSE][1]
    ankle_y = (pts[L_ANKLE][1] + pts[R_ANKLE][1]) / 2 if full_body else None
    span_px = (ankle_y - nose_y) if ankle_y else None

    vis = [lm[i].visibility for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    out["detected"] = True
    out["landmarks"] = {
        "shoulder_width_px": round(shoulder_px, 1),
        "hip_width_px": round(hip_px, 1),
        "torso_height_px": round(torso_px, 1),
        "full_body_visible": bool(full_body),
        "mean_core_visibility": round(float(np.mean(vis)), 2),
    }

    if not out["quality"]["resolution_ok"]:
        out["warnings"].append("Image is small; results improve with a larger photo (≥ 720px on the short side).")
    if not out["quality"]["sharp"]:
        out["warnings"].append("Image looks soft/blurry; use a sharper photo if possible.")
    if not out["quality"]["well_lit"]:
        out["warnings"].append("Lighting is extreme (very dark or blown out); try softer, even lighting.")

    # --- Optional measurement estimation (only with user-supplied height) ---
    if height_cm and full_body and span_px and span_px > 50:
        scale = height_cm / (span_px * 1.06)  # nose-to-ankle span is ~94% of height
        shoulder_cm = shoulder_px * 1.18 * scale   # landmarks sit inside the true biacromial width
        hip_cm = hip_px * 2.05 * scale             # hip landmarks are narrow vs hip circumference
        chest_cm = shoulder_cm * 2.35              # rough population ratio, clearly an estimate
        out["estimates"] = {
            "shoulder_cm": round(shoulder_cm, 1),
            "hip_circumference_cm": round(hip_cm, 1),
            "chest_circumference_cm": round(chest_cm, 1),
            "method": "MediaPipe pose landmarks scaled by user-provided height",
            "accuracy_note": ("Rough estimate from a single photo. Camera distance, angle, lens distortion, "
                              "clothing thickness and pose all affect the result. Do not use for medical or "
                              "made-to-measure purposes."),
        }
    else:
        out["estimates"] = None
        if height_cm and not full_body:
            out["warnings"].append("Measurement estimate needs a full-body photo (head to ankles).")

    # --- Overlay skeleton ---
    overlay = rgb.copy()
    mp_pose = _POSE
    with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                      min_detection_confidence=0.5) as pose2:
        res2 = pose2.process(rgb)
        if res2.pose_landmarks:
            _DRAW.draw_landmarks(
                overlay, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                _DRAW.DrawingSpec(color=(180, 83, 42), thickness=2, circle_radius=3),
                _DRAW.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            )
    from .image_utils import np_to_b64
    out["overlay_b64"] = np_to_b64(overlay)
    return out


def quality_check(rgb: np.ndarray) -> dict:
    """Cheap image-quality scan (blur/brightness/resolution) — no overlay rendering."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    h, w = rgb.shape[:2]
    return {
        "blur_variance": round(blur, 1),
        "brightness": round(bright, 1),
        "resolution_ok": bool(min(w, h) >= 360),
        "sharp": bool(blur >= 60),
        "well_lit": bool(30 <= bright <= 235),
    }


def pose_keypoints(rgb: np.ndarray) -> Optional[dict]:
    """Lightweight landmark extraction used by the fit-warp post-processor."""
    if not _AVAILABLE:
        return None
    h, w = rgb.shape[:2]
    with _POSE.Pose(static_image_mode=True, model_complexity=0,
                    min_detection_confidence=0.5) as pose:
        res = pose.process(rgb)
    if res.pose_landmarks is None:
        return None
    lm = res.pose_landmarks.landmark

    def pt(i):
        return (lm[i].x * w, lm[i].y * h, lm[i].visibility)

    sl, sr = pt(L_SHOULDER), pt(R_SHOULDER)
    hl, hr = pt(L_HIP), pt(R_HIP)
    ankle_y = max(pt(L_ANKLE)[1], pt(R_ANKLE)[1]) if max(pt(L_ANKLE)[2], pt(R_ANKLE)[2]) > 0.4 else None
    return {
        "center_x": float((sl[0] + sr[0]) / 2),
        "shoulder_y": float((sl[1] + sr[1]) / 2),
        "hip_y": float((hl[1] + hr[1]) / 2),
        "shoulder_width_px": float(abs(sr[0] - sl[0])),
        "hip_width_px": float(abs(hr[0] - hl[0])),
        "ankle_y": float(ankle_y) if ankle_y else None,
        "image_size": (int(w), int(h)),
    }
