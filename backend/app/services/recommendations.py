"""Colour-based outfit recommendations (rule-based, deliberately simple).

Secondary feature: practical pairing suggestions from the garment's dominant
colours and category. Not personalised styling advice.
"""
import colorsys


def _hex_to_hsv(hex_str: str):
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    return hh * 360, ss, vv


def _is_neutral(s: float, v: float) -> bool:
    return s < 0.18 or v < 0.14 or v > 0.92


_NEUTRAL_NAMES = [
    ("Charcoal", "#2F2F33"), ("Navy", "#1F2A44"), ("Khaki", "#B4A07A"),
    ("Olive", "#5B6236"), ("Ivory", "#F5F1E8"), ("Stone", "#CFC8BC"),
]


def recommend(dominant_colors: list[dict], category: str) -> dict:
    cols = dominant_colors or [{"hex": "#808080", "share": 1.0}]
    primary = cols[0]["hex"]
    h, s, v = _hex_to_hsv(primary)

    recs: list[str] = []
    pair_colors: list[str] = []

    if _is_neutral(s, v):
        recs.append("Neutral garment — almost any trouser works; choose contrast for colour pops on top or play it tonal.")
        if v < 0.3:
            pair_colors = ["#D8C3A5", "#8A8578", "#F5F1E8", "#B4432F"]
            recs.append("Pair with lighter neutrals (stone, ivory) or one muted accent (rust, olive).")
        else:
            pair_colors = ["#1F2A44", "#2F2F33", "#5B6236", "#B4432F"]
            recs.append("Pair with dark navy or charcoal to anchor the light base.")
    else:
        comp = (h + 180) % 360
        # Build two near-complementary accents as hex
        def hex_of(hh, ss, vv):
            r, g, b = colorsys.hsv_to_rgb(hh / 360, ss, vv)
            return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        pair_colors = [hex_of(comp, min(s + 0.1, 0.85), min(v, 0.75)),
                       hex_of((h + 30) % 360, min(s + 0.05, 0.8), min(v, 0.85)),
                       "#1F2A44", "#2F2F33"]
        recs.append(f"Dominant hue ≈ {h:.0f}°. Complementary accent ≈ {comp:.0f}° works for shoes, belt or layering pieces; keep everything else muted.")
        recs.append("Tonal dressing (same hue family, varied depth) also reads intentional and modern.")

    if category == "tops":
        bottoms = "Straight or tapered trousers in navy, charcoal, khaki or olive; dark denim for casual looks; avoid matching the exact garment colour."
        shoes = "Leather derbies/loafers (brown or oxblood) for smart looks; clean minimalist sneakers for casual. Match leather belt to shoes."
        accessories = "Simple watch, leather belt in matching tone, and one quiet accent (bracelet or pocket square) at most."
    else:
        bottoms = "Top it with a crisp white or light-neutral shirt, or a knit in the complementary accent from above."
        shoes = "White or off-white leather sneakers; brown/oxblood leather for smarter settings; avoid shoes louder than the garment."
        accessories = "Watch with leather strap, minimal belt; if the trouser has pattern, keep the top solid."

    return {
        "title": "Outfit pairing suggestions",
        "suggestions": [bottoms, shoes, accessories] + recs,
        "colors": pair_colors,
        "note": "Rule-based colour-theory suggestions from the garment's dominant colours. "
                "These are starting points, not rules — personal taste and dress codes win.",
    }
