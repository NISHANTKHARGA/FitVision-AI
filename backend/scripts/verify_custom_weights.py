"""Verify a (possibly user-trained) FASHN-architecture checkpoint.

Usage:
  python scripts/verify_custom_weights.py --weights-dir /path/to/your/checkpoint [--smoke]

Checks:
  1. Directory layout (model.safetensors + dwpose/*.onnx)
  2. safetensors header parses; tensor inventory printed
  3. Optional --smoke: builds the full pipeline and runs one real inference
     on the bundled sample images (CPU/GPU auto, ~1–2 min).

Exit code 0 = compatible, 1 = problem (read the messages).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

REQUIRED = ["model.safetensors", "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx"]


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="run one real inference to prove compatibility")
    ap.add_argument("--samples-dir", default=str(Path(__file__).resolve().parents[2] / "frontend" / "public" / "samples"))
    args = ap.parse_args()
    wdir = Path(args.weights_dir)
    ok = True

    print(f"Verifying checkpoint at: {wdir}")
    if not wdir.is_dir():
        print(f"  ✗ directory does not exist: {wdir}")
        return 1
    for rel in REQUIRED:
        f = wdir / rel
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            print(f"  ✓ {rel} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ MISSING: {rel}")
            ok = False
    if not ok:
        print("\nCopy the dwpose folder from the stock weights, and ensure your training run saved model.safetensors in the FASHN format.")
        return 1

    try:
        h = header(wdir / "model.safetensors")
        tensors = {k: v for k, v in h.items() if k != "__metadata__"}
        dtypes = {}
        for v in tensors.values():
            dtypes[v.get("dtype", "?")] = dtypes.get(v.get("dtype", "?"), 0) + 1
        n_params = "unknown"
        try:
            total = 0
            for v in tensors.values():
                shape = v.get("shape", [])
                num = 1
                for d in shape:
                    num *= d
                total += num
            n_params = f"{total / 1e6:.1f}M"
        except Exception:
            pass
        print(f"  ✓ safetensors parses: {len(tensors)} tensors · dtypes {dtypes} · ≈ {n_params} params")
    except Exception as exc:
        print(f"  ✗ could not parse model.safetensors header: {exc}")
        return 1

    if args.smoke:
        print("\nRunning smoke inference (4 steps)… this loads the full model.")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "src"))
            from fashn_vton import TryOnPipeline
            from PIL import Image
            import numpy as np
            pipe = TryOnPipeline(weights_dir=str(wdir))
            sdir = Path(args.samples_dir)
            person = Image.open(sdir / "sample-person.webp").convert("RGB")
            garment = Image.open(sdir / "sample-garment.webp").convert("RGB")
            out = pipe(person_image=person, garment_image=garment, category="tops",
                       num_timesteps=4, seed=42).images[0]
            dest = wdir / "smoke_test_output.png"
            out.save(dest)
            print(f"  ✓ smoke inference succeeded → {dest}")
        except Exception as exc:
            print(f"  ✗ smoke inference failed: {type(exc).__name__}: {exc}")
            print("    Common causes: wrong architecture (must be FASHN VTON v1.5 MMDiT), tensor name drift,")
            print("    dtype mismatch (model must be bfloat16-compatible), or VRAM exhaustion.")
            return 1

    print("\n✓ Checkpoint is compatible. Point FitVision at it with:  $env:FASHN_WEIGHTS = '<this dir>'  then restart the backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
