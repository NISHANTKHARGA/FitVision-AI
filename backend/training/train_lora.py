"""LoRA fine-tune of FASHN VTON v1.5 on VITON-HD (rectified-flow / v-prediction).

Run a quick smoke test first:
  python train_lora.py --data D:/virtual try on new/data/vitonhd --steps 30 --limit 64 --smoke

Full fine-tune example:
  python train_lora.py --data D:/virtual try on new/data/vitonhd --steps 4000 --out D:/virtual try on new/data/fitvision-lora

Writes a merged FASHN-format checkpoint + copied dwpose to --out (weights go to $FITVISION_WEIGHTS).
VITON-HD is CC BY-NC-SA 4.0 — research only. See docs/TRAINING.md.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "src"))

from fashn_vton.tryon_mmdit import TryOnModel  # noqa: E402
from fashn_vton.utils.checkpoint import load_checkpoint  # noqa: E402

from training.ckpt_forward import forward_ckpt  # noqa: E402
from training.dataset import VitonHDDataset, collate  # noqa: E402
from training.lora import inject_lora, freeze_base, lora_parameters, save_merged_checkpoint  # noqa: E402


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to extracted VITON-HD root")
    ap.add_argument("--out", default="D:/virtual try on new/data/fitvision-lora")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="limit dataset size (debug/smoke)")
    ap.add_argument("--smoke", action="store_true", help="tiny smoke run: few steps, eval skipped")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uncond-prob", type=float, default=0.1, help="CFG dropout probability")
    ap.add_argument("--resume", help="path to a previously merged checkpoint to start from (rarely needed; default = stock)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{dt.datetime.now():%H:%M:%S}] Loading TryOnModel (bf16)…")
    weights = args.resume or str(Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "weights" / "model.safetensors")
    model = TryOnModel()
    state = load_checkpoint(weights, device="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device=device, dtype=torch.bfloat16)
    model.train()

    replaced = inject_lora(model, rank=args.rank, alpha=2 * args.rank,
                           target_filters=("qkv", "proj", "mlp.0", "mlp.2", "linear1", "linear2"))
    freeze_base(model)
    n_train = sum(p.numel() for p in lora_parameters(model))
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected into {len(replaced)} linears · trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params")

    ds = VitonHDDataset(args.data, split="train", limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, collate_fn=collate, drop_last=True)
    print(f"Dataset: {len(ds)} pairs")

    opt = torch.optim.AdamW(lora_parameters(model), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    warmup = min(50, args.steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / max(warmup, 1)
        # cosine decay to 10% of peak
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress))).item())

    step = 0
    micro = 0
    t0 = time.time()
    accum_loss = 0.0
    accum_n = 0
    model.zero_grad(set_to_none=True)

    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            gt = batch["gt"].to(device, non_blocking=True)
            ca = batch["ca_image"].to(device, non_blocking=True)
            gi = batch["garment_image"].to(device, non_blocking=True)
            pp = batch["person_pose"].to(device, non_blocking=True)
            gp = batch["garment_pose"].to(device, non_blocking=True)
            cat = batch["category"].to(device, non_blocking=True)

            t = torch.rand(gt.shape[0], device=device)
            eps = torch.randn_like(gt)
            t_b = t[(...,) + (None,) * (gt.dim() - 1)]
            x_t = (1 - t_b) * gt + t_b * eps
            target = eps - gt

            # classifier-free dropout
            if torch.rand(1).item() < args.uncond_prob:
                mask = torch.zeros(gt.shape[0], dtype=torch.bool, device=device)
            else:
                mask = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = forward_ckpt(model, x_t, t, ca_images=ca, garment_images=gi,
                                    person_poses=pp, garment_poses=gp,
                                    mask=mask, garment_categories=cat)["x"]
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            accum_n += 1
            micro += 1

            if micro % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
                opt.step()
                model.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    avg = accum_loss / max(accum_n, 1)
                    print(f"[{dt.datetime.now():%H:%M:%S}] step {step}/{args.steps}  loss {avg:.5f}  ({(time.time()-t0):.0f}s elapsed)")
                    accum_loss, accum_n = 0.0, 0

    print(f"[{dt.datetime.now():%H:%M:%S}] Training done in {(time.time()-t0)/60:.1f} min. Saving merged checkpoint…")
    ckpt = out / "model.safetensors"
    save_merged_checkpoint(model, str(ckpt))
    # copy dwpose so the output dir is a valid FASHN_WEIGHTS dir
    dw_src = Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "weights" / "dwpose"
    dw_dst = out / "dwpose"
    if not dw_dst.exists():
        shutil.copytree(dw_src, dw_dst)
    (out / "training_info.json").write_text(json.dumps({
        "base": "FASHN VTON v1.5 (Apache-2.0)", "dataset": "VITON-HD (CC BY-NC-SA 4.0 — research only)",
        "steps": args.steps, "lora_rank": args.rank, "lr": args.lr,
        "label_suggestion": "FitVision VTON (VITON-HD LoRA) — research use only",
    }, indent=2))
    print(f"Checkpoint written to {out}")
    print("Point FitVision at it with:  $env:FASHN_WEIGHTS = '" + str(out) + "'  then restart the backend.")


if __name__ == "__main__":
    main()
