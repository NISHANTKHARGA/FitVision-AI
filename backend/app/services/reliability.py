"""Try-On Reliability Report.

Only *measured* indicators get numeric backing; everything else is an honest
qualitative assessment. No arbitrary confidence percentages.
"""
from typing import Optional

import cv2
import numpy as np

from .pose import pose_keypoints
from .preservation import garment_region_crop


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    # Lightweight SSIM implementation (no skimage dependency)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (7, 7), 1.5), cv2.GaussianBlur(b, (7, 7), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / ((mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2))
    return float(ssim_map.mean())


try:
    import mediapipe as mp
    _FACE_DET = mp.solutions.face_detection
    _FACE_AVAIL = True
except Exception:  # pragma: no cover
    _FACE_AVAIL = False


def _face_crop_detect(rgb: np.ndarray):
    """Scale-aware face crop via MediaPipe Face Detection. Returns None on failure."""
    if not _FACE_AVAIL:
        return None
    try:
        with _FACE_DET.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
            res = det.process(rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width)
        rbb = best.location_data.relative_bounding_box
        H, W = rgb.shape[:2]
        x, y, w, h = rbb.xmin, rbb.ymin, rbb.width, rbb.height
        # expand to include jaw/hair edge
        y = max(0.0, y - 0.12 * h)
        x = max(0.0, x - 0.08 * w)
        w *= 1.16
        h *= 1.28
        x1, y1 = int(x * W), int(y * H)
        x2, y2 = min(W - 1, int((x + w) * W)), min(H - 1, int((y + h) * H))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return rgb[y1:y2, x1:x2]
    except Exception:
        return None


def _face_crop(rgb: np.ndarray, kp_meta: Optional[dict]) -> np.ndarray:
    """Fallback: crop the head region using pose geometry (used only if face detection fails)."""
    det = _face_crop_detect(rgb)
    if det is not None:
        return det
    h, w = rgb.shape[:2]
    if kp_meta is None:
        return rgb[: h // 4]
    y_sh = kp_meta["shoulder_y"]
    cx = kp_meta["center_x"]
    half_w = kp_meta["shoulder_width_px"] * 0.38
    y1 = max(int(y_sh), 1)
    head_h = kp_meta["shoulder_width_px"] * 0.62
    y0 = max(0, int(y1 - head_h * 1.35))
    x0, x1 = max(0, int(cx - half_w)), min(w - 1, int(cx + half_w))
    return rgb[y0:y1, x0:x1]


def build_report(person_rgb: np.ndarray, result_rgb: np.ndarray,
                 pose_person: Optional[dict], pose_result: Optional[dict],
                 preservation: dict, measurements_provided: dict,
                 quality: dict) -> dict:
    items = []
    notes = [
        "'Measured' items are computed from the actual images. 'Qualitative' items are honest visual judgements.",
        "No confidence percentages are shown because diffusion-based try-on does not expose calibrated probabilities.",
    ]

    # 1. Person identity consistency — structural similarity of aligned face crops
    try:
        fa = cv2.cvtColor(_face_crop(person_rgb, pose_person), cv2.COLOR_RGB2GRAY)
        fb = cv2.cvtColor(_face_crop(result_rgb, pose_result or pose_person), cv2.COLOR_RGB2GRAY)
        if min(fa.shape[:2]) >= 20 and min(fb.shape[:2]) >= 20:
            SIZE = 128
            fa = cv2.resize(fa, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            fb = cv2.resize(fb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            s = _ssim_gray(fa, fb)
            strong = s >= 0.75
            items.append({"check": "Person identity consistency (face region SSIM)",
                          "status": "pass" if strong else "warning",
                          "measured": True,
                          "detail": f"Face-detection-aligned structural similarity (128×128): {s:.2f} "
                                    f"(≥0.75 = face appears intact). SSIM is not a biometric identity check."})
        else:
            items.append({"check": "Person identity consistency", "status": "unverified", "measured": False,
                          "detail": "Could not crop a usable face region from the images."})
    except Exception as exc:
        items.append({"check": "Person identity consistency", "status": "unverified", "measured": False,
                      "detail": f"Face-region comparison unavailable ({exc})."})

    # 2/3/4. Colour, pattern, logo/text — reuse measured preservation metrics
    pres = {i["aspect"]: i for i in preservation.get("items", [])}
    colour = pres.get("Colour similarity")
    if colour:
        items.append({"check": "Garment colour preservation",
                      "status": "pass" if colour["status"] == "preserved" else "warning",
                      "measured": True, "detail": colour["detail"]})
    pattern = pres.get("Pattern / print preservation")
    if pattern:
        items.append({"check": "Pattern preservation",
                      "status": "pass" if pattern["status"] == "preserved" else "warning",
                      "measured": True, "detail": pattern["detail"]})
    logo = pres.get("Logo / text preservation")
    if logo:
        items.append({"check": "Logo / text preservation",
                      "status": "pass" if logo["status"] == "preserved" else ("warning" if logo["status"] == "potentially_altered" else "unverified"),
                      "measured": logo.get("measured", False), "detail": logo["detail"]})

    # 5. Body-region alignment — pose detection on the result vs original
    if pose_person and pose_result:
        dx = abs(pose_result["center_x"] - pose_person["center_x"]) / max(pose_person["image_size"][0], 1)
        dsy = abs(pose_result["shoulder_y"] - pose_person["shoulder_y"]) / max(pose_person["image_size"][1], 1)
        dy = abs(pose_result["hip_y"] - pose_person["hip_y"]) / max(pose_person["image_size"][1], 1)
        drift = max(dx, dsy, dy)
        items.append({"check": "Body-region alignment (pose landmark drift)",
                      "status": "pass" if drift < 0.08 else ("warning" if drift < 0.15 else "warning"),
                      "measured": True,
                      "detail": f"Relative pose-landmark drift between input and result: {drift:.1%} of the frame "
                                f"(<8% = body geometry stable). Diffusion models re-render the whole frame, so small drift is expected."})
    else:
        items.append({"check": "Body-region alignment", "status": "unverified", "measured": False,
                      "detail": "Pose could not be detected in one of the images, so alignment is unverified."})

    # 6. Measurement availability
    body_ok = any(v for v in (measurements_provided.get("body") or {}).values())
    garment_ok = any(v for v in (measurements_provided.get("garment") or {}).values())
    if body_ok and garment_ok:
        items.append({"check": "Measurement availability", "status": "pass", "measured": True,
                      "detail": "Both body and garment measurements provided — fit analysis is measurement-based."})
    elif body_ok or garment_ok:
        items.append({"check": "Measurement availability", "status": "warning", "measured": True,
                      "detail": "Only one measurement set provided — fit analysis will be partial."})
    else:
        items.append({"check": "Measurement availability", "status": "unverified", "measured": False,
                      "detail": "No measurements provided — fit analysis is limited to visual comparison."})

    # 7. Areas of uncertainty (honest qualitative list)
    uncertainty = []
    if not quality.get("sharp", True):
        uncertainty.append("source images look soft; fine print/logo detail may be lost")
    if not quality.get("well_lit", True):
        uncertainty.append("extreme lighting can shift perceived colours")
    uncertainty.append("garment drape, folds and physical fit cannot be derived from a 2D diffusion render")
    uncertainty.append("hands, fingers and accessories near the garment edge are frequent artefact zones")
    uncertainty.append("long-to-short or bulky-to-slim garment swaps may leave traces of the original garment")
    items.append({"check": "Areas of uncertainty", "status": "warning", "measured": False,
                  "detail": "; ".join(uncertainty) + "."})

    return {"items": items, "notes": notes}
