# FitVision AI — Intelligent 2D Virtual Try-On & Fit Analysis

> **Your Style. Your Fit. Before You Buy.**
> Intelligent virtual try-on with AI-powered garment analysis and fit insights — running **locally** on your GPU.

FitVision AI is a working MVP that combines:

1. **AI Virtual Try-On Engine** — genuine garment-transfer diffusion model (FASHN VTON v1.5, Apache-2.0), not a paste-over.
2. **Fit Preference Simulator** — Slim / Regular / Oversized previews side by side (documented geometric approximation on top of the AI render — see `docs/FIT_WARP.md`).
3. **Body Measurement Assistant** — manual entry + optional pose-based estimation (MediaPipe; clearly labelled as a rough estimate).
4. **Garment Measurement Analysis** — transparent, configurable sizing rules produce plain-language verdicts per area. Never invented from photos.
5. **Garment Detail Preservation Checker** — colour ΔE, HSV histogram correlation, edge-density + ORB feature matching, CLIP ViT-B/32 embeddings and EasyOCR token overlap between the source garment and the render.
6. **Try-On Reliability Report** — measured indicators (face-region SSIM, pose-landmark drift, measurement availability) + honest qualitative notes. **No fabricated confidence percentages.**
7. **Outfit Recommendations** — rule-based colour-theory pairing suggestions (secondary feature).

## Hardware used & tested

| | |
|---|---|
| Machine | Intel Core i5-class CPU · 16 GB RAM · **NVIDIA RTX 4060 Laptop (8 GB)** |
| Diffusion render | **≈ 30–40 s** (Best 20 steps, default) · **≈ 13 s** (Fast 8 steps) · one render at a time (server-enforced GPU lock) |
| End-to-end (click → result screen) | **≈ 25–45 s** warm (model pre-warmed at startup) |
| Speed presets | **Draft** 4 steps g1.0 · **Fast** 8 g1.2 · **Balanced** 12 g1.5 · **Best** 20 g1.5 (default) |
| GPU memory | ~7.5 GB peak while rendering; closing other GPU apps keeps renders fast |
| Disk | ~4 GB for weights + caches; ~500 KB per result |
| CPU fallback | Possible in principle (float32) but impractically slow — not recommended, not auto-enabled |

## Quick start

### Prerequisites

- Node.js ≥ 20, Python 3.12 (tested) with pip
- (For real generation) NVIDIA GPU with ≥ 6 GB VRAM strongly recommended; CUDA 12.4 toolkit via PyTorch wheel

### 1 · Backend

```powershell
cd backend
python -m pip install -r requirements.txt

# FASHN VTON v1.5 is installed under ai/fashn-vton-1.5 (Apache-2.0)
cd ..\ai\fashn-vton-1.5
python -m pip install -e .
python scripts/download_weights.py --weights-dir ./weights   # ~2 GB, one time
cd ..\..

python -m uvicorn app.main:app --host 127.0.0.1 --port 8017
```

> **Important (Windows):** always start with `python -m uvicorn` — a bare `uvicorn`
> on this machine resolves to a different Python (3.14) without the ML dependencies.

### 2 · Frontend

```powershell
cd frontend
npm install
npm run build          # production build
npm run preview -- --port 5173
# or: npm run dev      # hot-reload development server
```

Open **http://localhost:5173** → *Start Virtual Try-On*.

### 3 · Try it without photos of your own

The Studio has a **“Try with sample photos”** button using bundled samples
(`frontend/public/samples/`). One click loads a person photo + a black FASHN tee image.

## Using your own trained model

FitVision AI can load **any checkpoint matching the FASHN VTON v1.5 architecture**
(your LoRA/continued fine-tune). Honest talk on what's realistic to train: see
**`docs/OWN_MODEL.md`**. The short version:

```powershell
# verify your checkpoint (structure, tensors, optional real smoke-inference)
python scripts/verify_custom_weights.py --weights-dir C:\path\to\your-checkpoint --smoke

# point the app at it and restart
$env:FASHN_WEIGHTS = "C:\path\to\your-checkpoint"
$env:FITVISION_MODEL_LABEL = "My Brand VTON v0.3 (LoRA)"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8017
```

The label shows in **Profile → Inference engine**. All analysis features (preservation,
reliability, fit rules) work unchanged on your model.

## API overview

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Engine status (model availability, steps default) |
| POST | `/api/analyze/person` | Pose detection, quality flags, pose overlay, optional height-scaled estimates |
| POST | `/api/analyze/garment` | Background removal (rembg), dominant colours, quality flags |
| POST | `/api/fit-analysis` | Standalone measurement → fit verdicts (transparent rules) |
| POST | `/api/tryon/generate` | Queues the try-on job (multipart). Returns `job_id` |
| GET | `/api/jobs/{id}` | Poll job status/progress/result |
| GET | `/api/results/{id}/image/{slim\|regular\|oversized}` | Rendered images |
| GET | `/api/results/{id}/source/{person\|garment}` | Input preview (for before/after) |
| GET/PUT | `/api/profile` | Local body-measurement profile (SQLite) |
| GET/DELETE | `/api/history` · `/api/history/{id}` | Saved results |
| POST | `/api/history/save/{id}` | Persist an ephemeral result to history |

Interactive API docs: **http://127.0.0.1:8017/docs**.

## What is real vs. approximated (read this before trusting output)

| Feature | Reality |
|---|---|
| Virtual try-on image | **Real** local diffusion inference (FASHN VTON v1.5). Not a mock, not an overlay. |
| Slim/Regular/Oversized comparison | **Approximation** — documented anatomy-anchored warp of the AI render. Not cloth physics. See `docs/FIT_WARP.md`. |
| Pose / skeleton / face-drift checks | **Measured** from the actual images (MediaPipe). |
| Garment colour check | **Measured** (HSV histogram correlation + dominant-colour CIE76 ΔE). |
| Pattern check | **Measured** (edge-density ratio + ORB matches + homography inliers). |
| Logo/text check | **Measured** if EasyOCR is installed and text is legible; otherwise reported as *not verified*. |
| Buttons/pockets/collar check | **Not verified** — below reliable automated resolution; reported honestly as such. |
| Fit verdicts (tight/ideal/…) | **Computed** from *your entered* measurements via editable rule constants in `fit_analysis.py`. Never from the photo. Never a fake "% match". |
| Measurement "estimation from photo" | **Rough heuristic** — needs user-provided height + full-body framing; labelled as such everywhere. |

## Project layout

```
├─ backend/            FastAPI + CV services + SQLite
│  ├─ app/routers/     API routes (tryon, profile)
│  ├─ app/services/    tryon_service, pose, garment, preservation, reliability,
│  │                   fit_analysis, fit_warp, recommendations, image_utils
│  ├─ tests/           pytest unit tests for pure logic (8 passing)
│  └─ data/            SQLite DB + ephemeral/saved results (auto-created)
├─ frontend/           React 19 + Vite + Tailwind v4
│  └─ src/pages/       Home · Studio · Results · Profile
├─ ai/fashn-vton-1.5/   Model code (pinned clone) + weights (~2 GB)
└─ docs/               FIT_WARP.md · MODEL_SELECTION.md · PRIVACY.md
```

## Testing

```powershell
cd backend
python -m pytest tests/ -q        # 8 unit tests (fit rules, warp, recommendations)
```

Manual E2E checklist (see `TESTING.md`): invalid uploads (wrong type, too big, corrupted),
missing garment image, empty measurements, GPU OOM handling, job polling,
history save/delete, 24 h ephemeral cleanup on restart.

## Licensing & credits

- App code: MIT (this repository).
- **FASHN VTON v1.5** — Apache-2.0 — © FASHN AI, https://github.com/fashn-AI/fashn-vton-1.5
- Sample person/garment images: bundled in the model repository under Apache-2.0.
- All analysis (MediaPipe, rembg, EasyOCR, CLIP) runs locally; no paid API is used anywhere.

## Limitations

See `docs/MODEL_SELECTION.md` (model) and in-app copy on every report. Short version:
diffusion renders are plausible images, not measurements; photo-based body estimates are
rough; 576×864 output; minutes-long render times on 8 GB GPUs; extreme garment shape
changes may leave traces of the original garment.
