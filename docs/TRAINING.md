> **Status (removed):** the VITON-HD dataset download was removed at the owner's request.
> The step-by-step below is kept as a reference for how a future fine-tune would be done
> if you re-download the dataset — but no training run is active and the app runs the
> stock FASHN model.

# Fine-Tuning FitVision's Model on VITON-HD (research use)

**You asked to train the model on the Zalando dataset. Here is the honest, working path.**
Short version of what you get, in order of effort:

| Goal | What it takes | What you get |
|---|---|---|
| “Better garment fidelity on studio shots” | LoRA fine-tune, ~2500 pairs, ~1–2 h on this GPU | Small but real improvement on VITON-HD-style inputs |
| “Good on everything” | Thousands of pairs, many hours, non-trivial tuning | Not plausible on 8 GB VRAM; don't promise it |
| “Train from scratch” | Millions of pairs, A100/H100 weeks | Not happening here |

**License warning (important, read first):** VITON-HD is **CC BY-NC-SA 4.0** — non-commercial
research use only. FASHN VTON v1.5 is Apache-2.0. A LoRA you train on VITON-HD inherits
the NC restriction: use it for research/evaluation, not in a commercial product, or swap
in a permissively-licensed dataset (or your own photos) for redistribution.

## What was built for you

- `backend/training/dataset.py` — VITON-HD loader producing the same tensors FASHN's
  inference pipeline feeds the model (rectified-flow training, conditional branch, 10%
  classifier-free dropout). Ground truth = the person wearing the garment. The garment
  is the flat-lay product shot; the person's upper body is masked out (that's exactly the
  task the model was pre-trained on).
- `scripts/cache_ca_images.py` — precomputes the clothing-agnostic person images with
  **FASHN's own parser** (multi-core). Do not skip this; the LIP colors shipped in the
  dataset are inconsistent across images, and guessing them trains garbage.
- `training/ckpt_forward.py` — gradient-checkpointed forward pass (same math as ai
  code, 11 GB → 5.3 GB VRAM).
- `training/lora.py` — LoRA injection (72 matrices, **7.7M trainable params**) + merge.
- `training/train_lora.py` — the trainer.
- `scripts/smoke_train.py` — quick “does the training step run?” test (no dataset needed).

## Quickstart

```powershell
cd backend
# 1. smoke the math on random tensors
python scripts/smoke_train.py

# 2. precompute agnostic images for N pairs (CPU-parallel; ~2.5 s CPU per pair)
python scripts/cache_ca_images.py --data D:/virtual try on new/data/vitonhd --limit 2500 --workers 6

# 3. tiny smoke train on the 96-pair dev subset (proves the loop end-to-end)
python training/train_lora.py --data D:/virtual try on new/data/vitonhd/smoke --steps 40 --limit 96

# 4. real fine-tune (adjust --steps to your patience; ~4–8 s/step on this GPU)
python training/train_lora.py --data D:/virtual try on new/data/vitonhd --steps 3000
```

Output: `D:/virtual try on new/data/fitvision-lora/model.safetensors` (merged, drop-in for
the FASHN weights) plus copied `dwpose/`. Then:

```powershell
$env:FASHN_WEIGHTS = "D:/virtual try on new/data/fitvision-lora"
$env:FITVISION_MODEL_LABEL = "FitVision VTON (VITON-HD LoRA) — research use only"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8017
```

The merged checkpoint is a **FASHN-format** drop-in — `verify_custom_weights.py` will
validate it, and `/api/health` will show your label.

## What “training” actually does here

The model was pre-trained as a rectified-flow (v-prediction) diffusion model on ~18M
pairs by FASHN. We continue training on VITON-HD pairs:

- `t ~ U(0,1)`, noise `ε`, target `v = ε − x0` (flow-matching objective)
- LoRA rank 16 on attention/MLP projections, bf16, AdamW lr 8e-5 cosine
- 10% classifier-free dropout keeps the unconditional branch usable at inference
- Gradient checkpointing keeps memory at ~5.3 GB; batch 1 × grad-accum 8

## Honest expectations

- Output gains concentrate on **Zalando-style studio shots** (the dataset's domain):
  cleaner garment transfer on flat frontal poses, slightly better small-print retention.
- It will **not** fix degraded hands, extreme poses, or improve non-Zalando photos.
  The stock FASHN model already scores ~0.85–0.87 CLIP cosine on our preservation checks;
  a small LoRA on ~2.5k pairs may buy you a few points there, sometimes zero.
- Evaluate with `/api` preservation metrics before/after at the same seed — same garment,
  same person, same prompt, 20 steps, g=1.5 — and keep the better model.
- **Do not ship a VITON-HD-trained checkpoint in a commercial product** without licensing
  review. For that, fine-tune on your own consented photos instead.

## Files produced by the full run

| File | Meaning |
|---|---|
| `fitvision-lora/model.safetensors` | Merged checkpoint (≈1.9 GB) — drop-in `FASHN_WEIGHTS` |
| `fitvision-lora/dwpose/` | Copied from stock weights (pose detector unchanged) |
| `fitvision-lora/training_info.json` | Provenance + label suggestion |

## Troubleshooting

- **OOM in smoke_train.py** — batch size is already 1; nothing to tune. If it OOMs on another
  machine, reduce LoRA rank (`--rank 8`) and re-run smoke.
- **Loss flat at ~2.0 for 100+ steps** — data pipeline is wrong (masks misaligned). Visualize
  `ds[i]` with `scripts/check_dataset.py` before training anything.
- **Result worse than stock after fine-tune** — expected if you overfit a narrow slice; use
  `--limit` and fewer `--steps`; always A/B with identical seeds.
- **Slow dataloading** — first pass over the dataset re-opens all images; use `--limit 500`
  while debugging, remove it for real runs.
