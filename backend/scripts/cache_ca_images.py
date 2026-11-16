"""Pre-cache clothing-agnostic (upper body) images for VITON-HD training.

Uses the EXACT same code the FASHN inference pipeline uses (FashnHumanParser +
create_clothing_agnostic_image) so the training distribution matches inference.

  python scripts/cache_ca_images.py --data D:/virtual try on new/data/vitonhd --limit 2500 --workers 6
"""
import argparse
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "src"))
from fashn_vton.preprocessing.agnostic import (  # noqa: E402
    create_clothing_agnostic_image, BODY_COVERAGE_TO_FASHN_LABELS, FASHN_LABELS_TO_IDS)
from fashn_vton.preprocessing.transforms import AspectPreserveResize  # noqa: E402

_parser = None


def worker_init():
    global _parser
    from fashn_human_parser import FashnHumanParser
    _parser = FashnHumanParser(device="cpu")


def _find_split_root(root: Path, split: str) -> Path:
    for cand in Path(root).rglob("image"):
        if cand.parent.name == split:
            return cand.parent
    raise FileNotFoundError(f"No {split}/image under {root}")


def _stems(split_dir: Path):
    stems = sorted(p.stem for p in (split_dir / "image").glob("*.jpg"))
    ok = set(stems) & {p.stem for p in (split_dir / "cloth").glob("*.jpg")}
    return [s for s in stems if s in ok]


def process_one(args):
    split_dir_s, stem, out_s, canvas = args
    split_dir = Path(split_dir_s)
    out = Path(out_s) / f"{stem}_ca.png"
    if out.exists():
        return (stem, "cached")
    person = Image.open(split_dir / "image" / f"{stem}.jpg").convert("RGB")
    pre = AspectPreserveResize(target_size=(canvas, canvas), mode="fit", backend="pil")
    person = pre(person, allow_upsampling=False)
    np_img = np.array(person)
    seg = _parser.predict(np_img)
    labels = BODY_COVERAGE_TO_FASHN_LABELS["upper"]
    labels_idx = [FASHN_LABELS_TO_IDS[l] for l in labels]
    ca = create_clothing_agnostic_image(
        img_np=np_img.copy(), seg_pred=seg.copy(),
        labels_to_segment_indices=labels_idx.copy(),
        body_coverage="upper", mask_limbs=True)
    Image.fromarray(ca).save(out)
    return (stem, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--canvas", type=int, default=864)
    args = ap.parse_args()

    root = Path(args.data)
    split_dir = _find_split_root(root, args.split)
    out_root = root / "processed" / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    stems = _stems(split_dir)
    if args.limit:
        stems = stems[:args.limit]
    print(f"Processing {len(stems)} pairs into {out_root}")

    t0 = time.time()
    with mp.Pool(args.workers, initializer=worker_init) as pool:
        for i, (stem, status) in enumerate(pool.imap_unordered(
                process_one, [(str(split_dir), s, str(out_root), args.canvas) for s in stems])):
            if (i + 1) % 50 == 0:
                print(f"[{time.time()-t0:.0f}s] {i+1}/{len(stems)} (latest: {stem})", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
