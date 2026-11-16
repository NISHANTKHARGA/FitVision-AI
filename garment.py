#Garment analysis:: background removal (rembg) and dominant colour extraction.

import io

import cv2
import numpy as np
from PIL import Image

_session = None


def _rembg_session():
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session("u2netp")  # lightweight model (~4 MB), much faster on CPU
    return _session


def remove_background(pil_img: Image.Image) -> Image.Image:
    from rembg import remove
    return remove(pil_img, session=_rembg_session())


def cutout(rgb: np.ndarray):
    """Background-removed garment as (rgb, alpha). Single computation shared by preview, metrics and analysis."""
    pil = Image.fromarray(rgb)
    out = remove_background(pil)  # RGBA
    arr = np.array(out)
    return arr[:, :, :3], arr[:, :, 3]


def dominant_colors(rgb: np.ndarray, mask: np.ndarray | None = None, k: int = 4) -> list[dict]:
    """Dominant colours via cv2.kmeans. mask: 255 = foreground."""
    if mask is not None:
        fg = rgb[mask > 127]
    else:
        fg = rgb.reshape(-1, 3)
    if len(fg) < 50:
        fg = rgb.reshape(-1, 3)
    sample = fg.astype(np.float32)
    if len(sample) > 12000:
        idx = np.random.choice(len(sample), 12000, replace=False)
        sample = sample[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(sample, k, None, criteria, 4, flags)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    total = counts.sum()
    out = []
    for i in order:
        c = centers[i]
        out.append({
            "hex": "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])),
            "rgb": [int(c[0]), int(c[1]), int(c[2])],
            "share": round(float(counts[i]) / total, 3),
        })
    return out


def analyze_garment(rgb: np.ndarray, cutout_parts: tuple | None = None) -> dict:
    pil = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    result = {
        "image_width": int(w),
        "image_height": int(h),
        "quality": {
            "blur_variance": round(blur, 1),
            "sharp": bool(blur >= 40),
            "resolution_ok": bool(min(w, h) >= 300),
        },
        "background_removed": False,
        "garment_png_b64": None,
        "dominant_colors": [],
        "orb_keypoints": 0,
        "warnings": [],
    }
    try:
        if cutout_parts is not None:
            cloth_rgb, alpha = cutout_parts
        else:
            cloth_rgb, alpha = cutout(rgb)
        cutout_pil = Image.fromarray(np.dstack([cloth_rgb, alpha]).astype(np.uint8), "RGBA")
        result["background_removed"] = True
        result["dominant_colors"] = dominant_colors(cloth_rgb, alpha, k=4)
        # Re-encode cutout to base64 for the frontend preview
        buf = io.BytesIO()
        cutout_pil.save(buf, format="PNG")
        import base64
        result["garment_png_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        gray_fg = cv2.cvtColor(cloth_rgb, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=800)
        kp, _ = orb.detectAndCompute(gray_fg, alpha)
        result["orb_keypoints"] = len(kp)
        fg_ratio = float((alpha > 127).mean())
        if fg_ratio > 0.92:
            result["warnings"].append("Garment fills almost the whole frame or has a non-uniform background; "
                                      "a product shot on a plain background gives the best try-on quality.")
    except Exception as exc:
        result["warnings"].append(f"Background removal unavailable ({exc}); colours analysed on the original image.")
        result["dominant_colors"] = dominant_colors(rgb, None, k=4)

    if not result["quality"]["resolution_ok"]:
        result["warnings"].append("Garment image is small; a larger image preserves detail better.")
    if not result["quality"]["sharp"]:
        result["warnings"].append("Garment image looks soft; prints and logos may be hard to preserve.")
    return result
