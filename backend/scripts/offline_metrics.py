import sys
sys.path.insert(0, r"D:\virtual try on new\backend")

import cv2
import numpy as np
from pathlib import Path

from app.services.preservation import garment_region_crop, check_preservation
from app.services.image_utils import load_image
from app.services.pose import pose_keypoints
from app.services.garment import cutout

SAMPLES = Path(r"D:\virtual try on new\frontend\public\samples")
RESULTS = Path(r"D:\virtual try on new\backend\data\results")

garment = load_image(SAMPLES / "sample-garment.webp")
cut_rgb, alpha = cutout(garment)
print("=== sample garment vs 4 presets (offline re-measure) ===")
for jid in ["3fdea9c0d72c", "6349d3108557", "1c888a9617d2", "f88c2963a7cb"]:
    res_path = RESULTS / jid / "tryon_regular.png"
    if not res_path.exists():
        print(jid, "missing")
        continue
    out_rgb = load_image(res_path)
    kp = pose_keypoints(out_rgb)
    res = check_preservation(garment, alpha, out_rgb, kp, "tops")
    tag = {"3fdea9c0d72c": "4 steps g1.0", "6349d3108557": "8 steps g1.2",
           "1c888a9617d2": "12 steps g1.5", "f88c2963a7cb": "20 steps g1.5"}[jid]
    print(f"\n--- {jid} ({tag}) ---")
    for i in res["items"]:
        print(f"  [{i['status'].ljust(18)}] {i['aspect']} :: {i['detail'][:120]}")
