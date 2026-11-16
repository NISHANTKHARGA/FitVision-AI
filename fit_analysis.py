#Fit analysis
"""Measurement-based fit analysis with transparent, configurable rules

No garment measurements are ever invented from the photo -- only user-entered
garment measurements are used. All rule constants live in FIT_RULES below and
are echoed back in the report so the logic is fully transparent.
"""
from typing import Optional

FIT_RULES = {
    "chest_ease_cm": {"slim": (2, 9), "regular": (9, 16), "oversized": (16, 30)},
    "waist_ease_cm": {"slim": (2, 8), "regular": (8, 15), "oversized": (15, 28)},
    "hip_ease_cm":   {"slim": (1, 7), "regular": (7, 14), "oversized": (14, 26)},
    "shoulder_delta_cm": {"slim": (-3.0, 1.0), "regular": (0.0, 3.0), "oversized": (2.0, 6.0)},
    # Garment length vs. body torso length preference (hip-skimming ~ optimal)
    "length_note": "Upper-body garment length is compared against torso length (scaled from height).",
}

UNITS_NOTE = "All eases in centimetres. Garment 'chest width'/'waist width' are flat half-widths and are doubled to compare with body circumferences."


def _verdict_from_interval(value: float, interval) -> str:
    lo, hi = interval
    if value < lo:
        return "tight"
    if value <= hi:
        return "ideal"
    if value <= hi + (hi - lo):
        return "relaxed"
    return "loose"


def analyze_fit(body: dict, garment: dict, category: str, fit: str) -> dict:
    verdicts = []
    fit = fit if fit in ("slim", "regular", "oversized") else "regular"

    def add(area, g, b, ease, status, note):
        verdicts.append({
            "area": area,
            "garment": round(g, 1) if g is not None else None,
            "body": round(b, 1) if b is not None else None,
            "ease_cm": round(ease, 1) if ease is not None else None,
            "status": status,
            "note": note,
        })

    height = body.get("height_cm")

    if category == "tops":
        if garment.get("chest_width_cm") and body.get("chest_cm"):
            g = garment["chest_width_cm"] * 2
            ease = g - body["chest_cm"]
            st = _verdict_from_interval(ease, FIT_RULES["chest_ease_cm"][fit])
            add("Chest", g, body["chest_cm"], ease, st,
                f"Garment chest ≈ {g:.0f} cm vs body {body['chest_cm']:.0f} cm → {ease:+.1f} cm ease ({fit} fit target {FIT_RULES['chest_ease_cm'][fit][0]}–{FIT_RULES['chest_ease_cm'][fit][1]} cm).")
        if garment.get("shoulder_cm") and body.get("shoulder_cm"):
            d = garment["shoulder_cm"] - body["shoulder_cm"]
            lo, hi = FIT_RULES["shoulder_delta_cm"][fit]
            st = "ideal" if lo <= d <= hi else ("tight" if d < lo else "relaxed")
            add("Shoulder", garment["shoulder_cm"], body["shoulder_cm"], d, st,
                f"Garment shoulder {garment['shoulder_cm']:.0f} cm vs body {body['shoulder_cm']:.0f} cm ({d:+.1f} cm). "
                f"Dropped shoulders are normal for oversized fits.")
        if garment.get("length_cm") and height:
            torso_est = height * 0.30
            delta = garment["length_cm"] - torso_est
            if abs(delta) <= 6:
                st, note = "ideal", f"Length {garment['length_cm']:.0f} cm is near the estimated torso length ({torso_est:.0f} cm); should sit at the hip."
            elif delta > 6:
                st, note = ("loose" if delta > 12 else "relaxed"), f"Length {garment['length_cm']:.0f} cm is longer than the estimated torso ({torso_est:.0f} cm); may bunch at the hip."
            else:
                st, note = ("tight" if delta < -12 else "relaxed"), f"Length {garment['length_cm']:.0f} cm is shorter than the estimated torso ({torso_est:.0f} cm); may ride up when arms are raised."
            add("Length", garment["length_cm"], round(torso_est, 1), delta, st, note)
        if garment.get("sleeve_cm"):
            add("Sleeve", garment["sleeve_cm"], None, None, "ideal" if garment["sleeve_cm"] >= 55 else "relaxed",
                "Sleeve length noted. Compare against your own arm (shoulder → wrist) for a personal check; no body arm measurement is stored.")
    else:  # bottoms
        if garment.get("waist_width_cm") and body.get("waist_cm"):
            g = garment["waist_width_cm"] * 2
            ease = g - body["waist_cm"]
            st = _verdict_from_interval(ease, FIT_RULES["waist_ease_cm"][fit])
            add("Waist", g, body["waist_cm"], ease, st,
                f"Garment waist ≈ {g:.0f} cm vs body {body['waist_cm']:.0f} cm → {ease:+.1f} cm ease ({fit} fit target {FIT_RULES['waist_ease_cm'][fit][0]}–{FIT_RULES['waist_ease_cm'][fit][1]} cm).")
        if garment.get("hip_width_cm") and body.get("hip_cm"):
            g = garment["hip_width_cm"] * 2
            ease = g - body["hip_cm"]
            st = _verdict_from_interval(ease, FIT_RULES["hip_ease_cm"][fit])
            add("Hip", g, body["hip_cm"], ease, st,
                f"Garment hip ≈ {g:.0f} cm vs body {body['hip_cm']:.0f} cm → {ease:+.1f} cm ease ({fit} fit target {FIT_RULES['hip_ease_cm'][fit][0]}–{FIT_RULES['hip_ease_cm'][fit][1]} cm).")
        if garment.get("length_cm"):
            add("Length", garment["length_cm"], None, None, "ideal",
                "Inseam/outseam preference is personal; verify against a garment you already own and like.")

    if not verdicts:
        return {
            "fit_preference": fit,
            "category": category,
            "verdicts": [],
            "overall": "Not enough measurements were provided to compare fit. Add at least chest (tops) or waist/hip (bottoms), plus shoulder width, for a useful analysis.",
            "recommended_fit": fit,
            "rules_used": FIT_RULES,
            "disclaimer": "Measurement-based estimate only. Different brands and cuts vary; always check the brand's size chart. " + UNITS_NOTE,
        }
#here
    statuses = [v["status"] for v in verdicts]
    tight_count = statuses.count("tight")
    loose_count = statuses.count("loose")

    if tight_count >= 2:
        overall = "Several areas look potentially tight in this size. Consider sizing up, or expect a close, body-hugging fit."
        recommended = "oversized" if fit == "regular" else "regular"
    elif tight_count == 1:
        overall = "Mostly workable, with one potentially tight area. Check the specific note above."
        recommended = fit
    elif loose_count >= 2:
        overall = "This size drapes generously. If you want a cleaner silhouette, consider sizing down."
        recommended = "slim" if fit == "regular" else "regular"
    else:
        overall = f"Overall, this matches a {fit} fit profile well based on the measurements provided."
        recommended = fit

    return {
        "fit_preference": fit,
        "category": category,
        "verdicts": verdicts,
        "overall": overall,
        "recommended_fit": recommended,
        "rules_used": FIT_RULES,
        "disclaimer": "Measurement-based estimate only. Different brands and cuts vary; always check the brand's size chart. " + UNITS_NOTE,
    }
