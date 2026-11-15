"""Fit-preference visual approximation via torso-anchored non-rigid warp.

IMPORTANT HONESTY NOTE
----------------------
The FASHN VTON v1.5 model has no explicit \"fit\" control. The slim / regular /
oversized previews shown by FitVision AI are produced by a *documented
geometric image-processing* step applied on top of the model's output:

* The garment region is identified with MediaPipe pose landmarks
  (shoulders -> hips for tops; hips -> ankles for bottoms).
* For every row y inside the garment band we compute a smooth scale profile
  s(y) = 1 + (target - 1) * w(y), where w(y) is a sine bump that is 0 at the
  band edges and 1 at mid-torso. Horizontal pixel remapping around the torso
  axis cx produces a true non-uniform warp (shoulders less affected than the
  chest/waist, unlike a plain resize).
* Band edges are feather-blended with the unmodified pixels so seams are
  invisible.

This preserves the garment's pixel identity (colours, prints, logos) because
nothing is redrawn -- only resampled. It is a *visual approximation* of
silhouette, NOT a prediction of physical fit, drape or garment physics.
"""
import cv2
import numpy as np

# Horizontal garment-width scale at mid-torso per fit preference
FIT_SCALES = {"slim": 0.92, "regular": 1.0, "oversized": 1.09}
# Oversized garments also hang lower; add a slight vertical stretch
FIT_VSCALES = {"slim": 0.99, "regular": 1.0, "oversized": 1.03}


def _band(kp: dict, category: str):
    h = kp["image_size"][1]
    torso = max(kp["hip_y"] - kp["shoulder_y"], 1.0)
    if category == "tops":
        top = kp["shoulder_y"] - 0.28 * torso
        bottom = kp["hip_y"] + 0.45 * torso
    else:  # bottoms
        top = kp["hip_y"] - 0.08 * torso
        foot = kp["ankle_y"] if kp.get("ankle_y") else float(h - 1)
        bottom = min(float(h - 1), foot + 0.05 * torso)
    top = max(0.0, top)
    bottom = min(float(h - 1), bottom)
    return top, bottom


def fit_warp(rgb: np.ndarray, kp: dict | None, category: str, fit: str) -> np.ndarray:
    if fit == "regular" or kp is None:
        return rgb.copy()
    target = FIT_SCALES[fit]
    vscale = FIT_VSCALES[fit]
    h, w = rgb.shape[:2]
    band_top, band_bottom = _band(kp, category)
    if band_bottom - band_top < 30:
        return rgb.copy()

    cx = kp["center_x"]
    band_h = band_bottom - band_top

    # Precompute per-row horizontal scale and the feather blend weight
    rows = np.arange(h, dtype=np.float32)
    in_band = (rows >= band_top) & (rows <= band_bottom)
    t = np.zeros(h, dtype=np.float32)
    t[in_band] = (rows[in_band] - band_top) / max(band_h, 1.0)
    wobble = np.clip(np.sin(np.pi * np.clip(t, 0, 1)), 0, None)
    bump = wobble ** 0.85
    scale = np.where(in_band, 1.0 + (target - 1.0) * bump, 1.0).astype(np.float32)

    # Vertical stretch about the band centre (mostly affects oversized)
    cy = (band_top + band_bottom) / 2.0
    src_y = np.where(in_band, cy + (rows - cy) / vscale, rows).astype(np.float32)

    cols = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    map_x = (cx + (cols - cx) / scale[:, None]).astype(np.float32)
    map_y = src_y[:, None].repeat(w, axis=1).astype(np.float32)

    warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

    # Feather-blend the band edges with the original image
    feather = np.zeros(h, dtype=np.float32)
    edge = max(int(0.12 * band_h), 8)
    feather[in_band] = 1.0
    for y in range(int(band_top), min(h, int(band_top) + edge)):
        feather[y] = (y - band_top) / edge
    for y in range(max(0, int(band_bottom) - edge), min(h, int(band_bottom) + 1)):
        feather[y] = (band_bottom - y) / edge
    feather = np.clip(feather, 0, 1)[:, None, None]
    out = warped * feather + rgb * (1 - feather)
    return out.astype(np.uint8)
