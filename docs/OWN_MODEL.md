# Using Your Own Trained Model

Short version: FitVision AI can run against **any checkpoint that matches the
FASHN VTON v1.5 architecture** (MMDiT, pixel-space, 12×12 patches, 8 double-stream +
16 single-stream blocks, ~972M params, bfloat16, 576×864). You point the backend at
it with one environment variable — no code changes.

> **Honesty note first.** Training a competitive try-on model from scratch is a
> research-scale effort (millions of person/garment/ground-truth triplets, many GPU
> weeks, FASHN's own recipe uses ≥18M pairs). On one RTX 4060 that is not realistic.
> What *is* realistic is **continuing training / LoRA-style fine-tuning** of the FASHN
> architecture on your own data — and FASHN ships **inference-only code, no trainer**.
> So today, "your own trained model" means: you (or someone) trains a checkpoint in the
> FASHN architecture; FitVision loads it.

## What FitVision needs to load it

```
your-checkpoint/
├── model.safetensors          # your trained FASHN-architecture weights (any dtype the pipeline can cast; bf16 recommended)
└── dwpose/
    ├── yolox_l.onnx           # pose detector (ships with FASHN; copy it)
    └── dw-ll_ucoco_384.onnx   # DWPose body model (ships with FASHN; copy it)
```

The human-parser weights (~244 MB) are auto-downloaded by HuggingFace on first use and
cache under `HF_HOME`.

## Step 1 — Train / fine-tune (the hard part you must supply)

Realistic options, in increasing difficulty:

1. **Continued fine-tuning of FASHN on your paired data.** Dataset: triplets of
   `person.webp` / `garment.webp` / `ground_truth.webp` (the person wearing the
   garment). You need the FASHN training architecture (not released) — you'd re-implement
   their two-phase masked+synthetic recipe or fine-tune with a diffusion LoRA stack
   (e.g. diffusers `TrainDiffusionPipeline`-style) on their public weights.
   - LoRA/DoRA on the 972M MMDiT: trainable ≈ 1–3% of params; fits in 8 GB with gradient
     checkpointing, small batch, bf16, 512–864 px.
   - Expect days per epoch on ~10k pairs; keep `dwpose` fixed and train only the try-on
     transformer.
2. **Full continue-training** — needs a datacenter-class GPU (A100/H100), their exact
   data recipe (≥18M pairs for phase 1). Not feasible on a single 4060.
3. **A completely different architecture** (CatVTON, IDM-VTON) — these are
   **CC BY-NC-SA 4.0 (non-commercial)**; swappable in principle but, like FASHN,
   they need their own preprocessing stack. Not drop-in.

FitVision does **not** ship a fake "train your own model" button. Anyone selling you
one-click training of a photorealistic try-on model on 8 GB is overselling.

## Step 2 — Verify your checkpoint

```powershell
cd backend
python scripts/verify_custom_weights.py --weights-dir C:\path\to\your-checkpoint
# deeper check (loads the model, runs one real 4-step inference):
python scripts/verify_custom_weights.py --weights-dir C:\path\to\your-checkpoint --smoke
```

The verifier prints tensor inventory and either passes (exit 0) or tells you exactly
what's wrong (missing files, parse failure, architecture/dtype mismatch).

## Step 3 — Point FitVision at it

```powershell
$env:FASHN_WEIGHTS = "C:\path\to\your-checkpoint"
$env:FITVISION_MODEL_LABEL = "Acme Brand VTON v0.3 (LoRA, 12k pairs)"   # shown in the UI
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8017
```

- Model label appears in **Profile → Inference engine** and in `/api/health`.
- Everything else (preservation checks, fit analysis, reports) works unchanged — they
  compare images, not your model's lineage.
- If your checkpoint is worse than stock FASHN, you'll see it immediately in the same
  preservation/reliability metrics. That's the honesty layer doing its job.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `✗ could not parse model.safetensors` | File truncated (partial copy) or not a safetensors file |
| `missing keys` / `size mismatch` in smoke test | Architecture/tensor-name drift — your trainer renamed layers |
| CUDA OOM in smoke test | Trainable merge must stay bf16; check `model.config` dtypes |
| Output is grey sludge | LR too high / too many epochs — your checkpoint overwrote the prior |
