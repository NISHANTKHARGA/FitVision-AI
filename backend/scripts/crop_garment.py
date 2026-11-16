import os
from PIL import Image, ImageDraw, ImageFont

tmp = os.environ["TEMP"]
RESULTS = r"D:\virtual try on new\backend\data\results"
tiles = []
names = [("SOURCE GARMENT", r"D:\virtual try on new\frontend\public\samples\sample-garment.webp")]
for jid, label in [("3fdea9c0d72c", "4 STEPS · g1.0"), ("6349d3108557", "8 STEPS · g1.2"),
                   ("1c888a9617d2", "12 STEPS · g1.5"), ("f88c2963a7cb", "20 STEPS · g1.5")]:
    tiles.append((label, os.path.join(RESULTS, jid, "tryon_regular.png")))

panels = []
for label, path in tiles:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    crop = im.crop((int(w * 0.30), int(h * 0.13), int(w * 0.82), int(h * 0.45)))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    panels.append((label, crop))

# Source garment as first panel (fitted to same height)
src = Image.open(tiles and r"D:\virtual try on new\frontend\public\samples\sample-garment.webp").convert("RGB")
src = src.resize((int(src.width * 0.75), int(src.height * 0.75)), Image.LANCZOS)
panels.insert(0, ("SOURCE GARMENT", src))

tw = max(p.width for _, p in panels)
th = max(p.height for _, p in panels)
pad, label_h = 16, 34
sheet = Image.new("RGB", (tw * len(panels) + pad * (len(panels) + 1), th + label_h + 2 * pad), (250, 248, 244))
d = ImageDraw.Draw(sheet)
x = pad
for label, p in panels:
    sheet.paste(p, (x, pad + label_h))
    d.text((x + 2, pad + 6), label, fill=(23, 21, 18))
    x += p.width + pad
sheet.save(os.path.join(tmp, "garment_vs_render.png"))
print("saved", sheet.size)
