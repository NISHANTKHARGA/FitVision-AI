import os
from PIL import Image

tmp = os.environ["TEMP"]
for n in ["a4", "a8", "a12", "a20"]:
    im = Image.open(os.path.join(tmp, n + ".png"))
    w, h = im.size
    crop = im.crop((int(w * 0.30), int(h * 0.15), int(w * 0.80), int(h * 0.45)))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    crop.save(os.path.join(tmp, n + "_chest.png"))
    print(n, im.size)

# Contact sheet
tiles = [Image.open(os.path.join(tmp, n + "_chest.png")) for n in ["a4", "a8", "a12", "a20"]]
tw = max(t.width for t in tiles)
th = max(t.height for t in tiles)
sheet = Image.new("RGB", (tw * 2 + 30, th * 2 + 60), (250, 248, 244))
for i, t in enumerate(tiles):
    x = (i % 2) * (tw + 30)
    y = (i // 2) * (th + 30)
    sheet.paste(t, (x, y))
sheet.save(os.path.join(tmp, "chest_contact.png"))
print("saved chest_contact.png")
