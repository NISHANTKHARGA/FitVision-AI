#Fifth
"""Garment Detail Preservation Checker.
Compares the uploaded garment photo against the garment region of the
generated try-on result. Every check reports whether it is *measured*
(an actual number from the images) or *not_verified* (the model cannot
support the claim). No confidence percentages are invented.

Methods:
* Colour similarity  — HSV histogram correlation + dominant-colour CIE76 ΔE
* Pattern preservation — Canny edge-density ratio + ORB feature matching
* Embedding similarity — CLIP ViT-B/32 cosine similarity (CPU)
* Logo / text — EasyOCR token overlap between garment and result (CPU)
"""
import re
from typing import Optional

import cv2
import numpy as np

_CLIP = None
_CLIP_PROC = None
_OCR = None


def _clip():
    global _CLIP, _CLIP_PROC
    if _CLIP is None:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        _CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP.eval()
    return _CLIP, _CLIP_PROC


def _ocr_reader():
    global _OCR
    if _OCR is None:
        import easyocr
        _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR


def garment_region_crop(rgb: np.ndarray, kp: dict, category: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    if kp is None:
        # Fallback: central third of the image
        return rgb[h // 5: int(h * 0.75), w // 5: int(w * 0.8)]
    cx = kp["center_x"]
    half_w = max(kp["shoulder_width_px"], kp["hip_width_px"]) * 0.95
    torso = max(kp["hip_y"] - kp["shoulder_y"], 1.0)
    if category == "tops":
        y0, y1 = kp["shoulder_y"] - 0.25 * torso, kp["hip_y"] + 0.40 * torso
    else:
        y0, y1 = kp["hip_y"] - 0.08 * torso, (kp["ankle_y"] if kp.get("ankle_y") else h - 1)
    x0, x1 = int(cx - half_w), int(cx + half_w)
    y0, y1 = max(0, int(y0)), min(h - 1, int(y1))
    x0, x1 = max(0, x0), min(w - 1, x1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return rgb[h // 5: int(h * 0.75), w // 5: int(w * 0.8)]
    return rgb[y0:y1, x0:x1]


def _lab_hist(rgb: np.ndarray, mask: np.ndarray | None = None):
    """Lighting-invariant chroma histogram (a,b channels of Lab)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    m = None
    if mask is not None:
        m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    hist = cv2.calcHist([lab], [1, 2], m, [24, 24], [0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _hsv_hist(rgb: np.ndarray, mask: np.ndarray | None = None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = None
    if mask is not None:
        m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    hist = cv2.calcHist([hsv], [0, 1], m, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _dominant_lab(rgb: np.ndarray, k: int = 3, mask: np.ndarray | None = None):
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if mask is not None:
        pts = lab_img[mask > 127]
    else:
        pts = lab_img.reshape(-1, 3)
    if len(pts) < 50:
        pts = lab_img.reshape(-1, 3)
    lab = pts.astype(np.float32)
    if len(lab) > 8000:
        idx = np.random.RandomState(0).choice(len(lab), 8000, replace=False)
        lab = lab[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
    _, _, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers


def _ciede2000_approx(lab1, lab2):
    """Greedy nearest-neighbour matching between dominant clusters (order-independent)."""
    remaining = list(lab2)
    errs = []
    for l in lab1:
        if not remaining:
            break
        d = np.sqrt(np.sum((np.array(remaining) - l) ** 2, axis=1))
        i = int(np.argmin(d))
        errs.append(float(d[i]))
        remaining.pop(i)
    return float(np.mean(errs)) if errs else 255.0


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def _orb_matches(g: np.ndarray, r: np.ndarray, gmask: np.ndarray | None = None) -> dict:
    orb = cv2.ORB_create(nfeatures=1200)
    kp1, des1 = orb.detectAndCompute(g, gmask)
    kp2, des2 = orb.detectAndCompute(r, None)
    if des1 is None or des2 is None or len(des1) < 12 or len(des2) < 12:
        return {"matches": 0, "good_matches": 0, "ratio": 0.0, "inlier_ratio": None}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < 0.78 * n.distance]
    inlier_ratio = None
    if len(good) >= 8:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_ratio = float(mask.sum() / max(len(mask), 1))
    return {"matches": int(len(knn)), "good_matches": int(len(good)),
            "ratio": round(len(good) / max(len(knn), 1), 3),
            "inlier_ratio": round(inlier_ratio, 3) if inlier_ratio is not None else None}


def _clip_similarity(a: np.ndarray, b: np.ndarray) -> float:
    from PIL import Image
    import torch
    model, proc = _clip()
    ia = Image.fromarray(a)
    ib = Image.fromarray(b)
    with torch.no_grad():
        emb = []
        for im in (ia, ib):
            inp = proc(images=im, return_tensors="pt")
            feat = model.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            emb.append(feat)
        sim = float(torch.mm(emb[0], emb[1].T).item())
    return sim


_TOKEN_RE = re.compile(r"[A-Za-z0-9&'’-]{2,}")


def _ocr_tokens(reader, rgb: np.ndarray) -> list[str]:
    try:
        out = reader.readtext(rgb, detail=0, paragraph=False)
    except Exception:
        return []
    toks = []
    for chunk in out:
        toks.extend(_TOKEN_RE.findall(str(chunk)))
    return [t.upper() for t in toks]


def _token_overlap(a: list[str], b: list[str]) -> dict:
    sa, sb = set(a), set(b)

    def _fuzzy(tok: str) -> bool:
        for other in sb:
            # substring-tolerant: OCR tends to drop trailing glyphs
            if tok in other or other in tok:
                return True
        return False

    if not sa:
        return {"garment_tokens": [], "result_tokens": sorted(sb), "overlap": [], "recall": None}
    overlap = sorted(t for t in sa if _fuzzy(t))
    return {"garment_tokens": sorted(sa), "result_tokens": sorted(sb),
            "overlap": overlap, "recall": round(len(overlap) / len(sa), 3)}


def check_preservation(garment_rgb: np.ndarray, garment_mask: Optional[np.ndarray],
                       result_rgb: np.ndarray, kp: Optional[dict], category: str) -> dict:
    region = garment_region_crop(result_rgb, kp, category)
    items = []
    gmask = garment_mask if garment_mask is not None and garment_mask.sum() > 0 else None
    if gmask is not None:
        gclip = np.where(gmask[..., None] > 127, garment_rgb, 255).astype(np.uint8)
        g_gray_m = (g_gray_full := cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2GRAY)) * (gmask > 127)
    else:
        gclip = garment_rgb
        g_gray_m = cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2GRAY)
    notes = [
        "Metrics compare the uploaded garment photo with the garment region of the generated image.",
        "Flat-lay vs worn garments differ in perspective, folds and lighting; low feature matches do not necessarily mean the design changed.",
        "Measured metrics are shown with raw values. Anything the pipeline cannot verify is explicitly listed as 'not verified'.",
    ]

    # ---------- 1. Colour ----------
    g_hist = _hsv_hist(garment_rgb, gmask)
    r_hist = _hsv_hist(region)
    hist_corr = float(cv2.compareHist(g_hist, r_hist, cv2.HISTCMP_CORREL))
    g_chroma = _lab_hist(garment_rgb, gmask)
    r_chroma = _lab_hist(region)
    chroma_corr = float(cv2.compareHist(g_chroma, r_chroma, cv2.HISTCMP_CORREL))
    g_dom = _dominant_lab(garment_rgb, 3, gmask)
    r_dom = _dominant_lab(region)
    dE = _ciede2000_approx(g_dom, r_dom)
    # Lighting changes shift V and L more than chroma; weight chroma correlation accordingly
    color_ok = (chroma_corr >= 0.45 and dE <= 45) or (hist_corr >= 0.6 and dE <= 40)
    items.append({
        "aspect": "Colour similarity",
        "status": "preserved" if color_ok else "potentially_altered",
        "measured": True,
        "detail": (f"Chroma (Lab a/b) histogram correlation {chroma_corr:.2f} · HSV histogram correlation {hist_corr:.2f} · "
                   f"dominant-colour CIE76 ΔE {dE:.1f} (lower is better)."),
        "metric": {"chroma_correlation": round(chroma_corr, 3), "histogram_correlation": round(hist_corr, 3),
                   "dominant_delta_e": round(dE, 1)},
    })

    # ---------- 2. Pattern ----------
    g_gray = g_gray_m
    r_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    g_ed, r_ed = _edge_density(g_gray), _edge_density(r_gray)
    density_ratio = (r_ed / g_ed) if g_ed > 1e-4 else None
    orb = _orb_matches(g_gray, r_gray, gmask if gmask is not None else None)
    pattern_ok = (density_ratio is not None and 0.45 <= density_ratio <= 2.2) or orb["good_matches"] >= 25
    items.append({
        "aspect": "Pattern / print preservation",
        "status": "preserved" if pattern_ok else "potentially_altered",
        "measured": True,
        "detail": (f"Edge density garment {g_ed:.3f} vs result {r_ed:.3f}"
                   + (f" (ratio {density_ratio:.2f})." if density_ratio is not None else ".")
                   + f" ORB good matches: {orb['good_matches']}"
                   + (f", homography inliers: {orb['inlier_ratio']:.0%}." if orb.get("inlier_ratio") is not None else ".")),
        "metric": {"edge_density_garment": round(g_ed, 4), "edge_density_result": round(r_ed, 4),
                   "density_ratio": round(density_ratio, 3) if density_ratio is not None else None,
                   **orb},
    })

    # ---------- 3. Embedding (CLIP) ----------
    try:
        sim = _clip_similarity(gclip, region)
        items.append({
            "aspect": "Overall garment appearance (CLIP ViT-B/32)",
            "status": "preserved" if sim >= 0.75 else ("potentially_altered" if sim < 0.60 else "preserved"),
            "measured": True,
            "detail": f"CLIP image-embedding cosine similarity {sim:.2f} (1 = identical semantics/appearance). "
                      "Treadmill of different poses and folds typically lands 0.7–0.95.",
            "metric": {"clip_cosine": round(sim, 3)},
        })
    except Exception as exc:
        items.append({
            "aspect": "Overall garment appearance (CLIP)",
            "status": "not_verified",
            "measured": False,
            "detail": f"CLIP embedding comparison unavailable ({exc}).",
            "metric": None,
        })

    # ---------- 4. Logo / text ----------
    try:
        reader = _ocr_reader()
        g_tokens = _ocr_tokens(reader, garment_rgb)
        r_tokens = _ocr_tokens(reader, region)
        ov = _token_overlap(g_tokens, r_tokens)
        if not g_tokens:
            items.append({
                "aspect": "Logo / text preservation",
                "status": "not_verified",
                "measured": True,
                "detail": "No legible text detected in the garment photo, so text/logo preservation cannot be confirmed. "
                          "If the garment has a logo, check it visually in the result.",
                "metric": ov,
            })
        else:
            ok = (ov["recall"] or 0) >= 0.6
            items.append({
                "aspect": "Logo / text preservation",
                "status": "preserved" if ok else "potentially_altered",
                "measured": True,
                "detail": f"Garment text tokens: {ov['garment_tokens'] or '∅'}; result tokens: {ov['result_tokens'] or '∅'}; "
                          f"token recall {ov['recall']}. OCR on garments is noisy — verify visually.",
                "metric": ov,
            })
    except Exception as exc:
        items.append({
            "aspect": "Logo / text preservation",
            "status": "not_verified",
            "measured": False,
            "detail": f"OCR unavailable ({exc}).",
            "metric": None,
        })

    # ---------- 5. Shape / structure ----------
    items.append({
        "aspect": "Buttons, pockets, collar, sleeves (shape & structure)",
        "status": "not_verified",
        "measured": False,
        "detail": "Fine structural details (buttons, buttonholes, pocket seams, collar ribbing) fall below the reliable "
                  "resolution of automated comparison. Inspect the generated image closely against the garment photo.",
        "metric": None,
    })

    preserved = sum(1 for i in items if i["status"] == "preserved")
    altered = sum(1 for i in items if i["status"] == "potentially_altered")
    summary = (f"{preserved} check(s) preserved, {altered} potentially altered, "
               f"{len(items) - preserved - altered} not verified. "
               "This is a decision-support indicator, not proof of design fidelity.")
    return {"items": items, "summary": summary, "method_notes": notes}
