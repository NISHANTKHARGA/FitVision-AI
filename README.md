# FitVision AI

> **Your Style. Your Fit. Before You Buy.**
> Intelligent 2D virtual try-on with AI-powered garment transfer, honest fit analysis and garment-preservation verification â€” all running **locally**, not in the cloud.

[![React](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-171512?logo=react&logoColor=61DAFB)](frontend)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-171512?logo=fastapi&logoColor=009688)](backend)
[![Model](https://img.shields.io/badge/try--on%20model-FASHN%20VTON%20v1.5-B4432F)](ai/fashn-vton-1.5)
[![GPU](https://img.shields.io/badge/GPU-RTX%204060%20%C2%B7%208%20GB-171512)](#performance)
[![Tests](https://img.shields.io/badge/tests-8%20passing-55603F)](backend/tests)
[![License](https://img.shields.io/badge/MIT-blue)](#license--credits)

<p align="center">
  <img src="docs/assets/fig_before_after.png" alt="FitVision AI â€” before and after virtual try-on" width="100%">
</p>

<p align="center">
  <a href="https://fitvision-ai-cyan.vercel.app">
    <img src="docs/assets/fig_vercel_button.png" alt="Launch FitVision AI â€” live demo on Vercel" width="420">
  </a>
</p>

## â–¶ Live demo (Vercel)

**https://fitvision-ai-cyan.vercel.app** â€” the full UI is hosted on Vercel.

The try-on model itself needs a local GPU, so the hosted page talks to the backend running on your own machine:

1. **On this PC** (the one it was set up on): make sure the backend is running â€”
   `cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8017` â€”
   then open the link and everything works, including generation.
2. **On any other machine**: you need the backend there too (same steps, ~2 GB one-time weight download, NVIDIA GPU â‰¥ 6 GB VRAM recommended). The hosted UI alone cannot render â€” there is no GPU in the cloud for free.
3. **First generation after a cold start** waits ~40 s for model load; subsequent renders are seconds.

Why not the whole stack on Vercel? Serverless functions have no GPU and ~10â€“60 s wall-clock limits; FASHN VTON v1.5 needs ~8 GB VRAM and a real GPU â€” Vercel hosts the frontend only. The backend CORS is pre-configured for this exact Vercel origin (`FITVISION_CORS_ORIGINS` env var changes it).

## A real run, end to end

Everything below is from one actual user session (uploaded photos â†’ processing â†’ result), not a mock-up:

**1 Â· Input â€” your photo and the garment**

<p align="center">
  <img src="docs/assets/fig_input.png" alt="FitVision AI input: user photo and garment photo" width="100%">
</p>

**2 Â· Processing â€” what the CV stack extracts**

<p align="center">
  <img src="docs/assets/fig_processing.png" alt="FitVision AI processing: pose detection overlay, garment background removal and dominant-colour palette" width="100%">
</p>

**3 Â· Output â€” the generated try-on**

<p align="center">
  <img src="docs/assets/fig_output.png" alt="FitVision AI output: before and after the AI try-on" width="100%">
</p>

---

## The problem we're solving â€” and the context

Online fashion has a **fidelity gap**. A product page shows a garment on a model you don't resemble, in a pose you don't hold, in lighting you'll never see. Buyers are left guessing three questions that a purchase hinges on:

1. **How will it look on *me*?** â€” body shape, skin undertone, proportions
2. **How will it *fit*?** â€” will it be tight at the chest, drown me at the waist, ride up at the hem?
3. **Is this the *same design*?** â€” did the print survive? The logo? The collar? Or did the AI "reinterpret" it?

Existing free options fail differently:

| Approach | What it does | Why it fails |
|---|---|---|
| Image-paste "try-on" apps | Overlay a flat PNG on a photo | No drape, no perspective, face-of-garment artefacts, obviously fake |
| Text-to-image generators | "person wearing red dress" | Ignores the *actual* garment; hallucinates a different one |
| Commercial try-on APIs | Paid, cloud-hosted | Cost per call, latency, privacy of your photos |
| Research models unmodified | VITON-HD pipelines | Masks, multi-stage pre/post-processing, no fit reasoning, no honesty layer |

**FitVision AI** closes that gap with a genuine garment-transfer diffusion model (FASHN VTON v1.5), wrapped in an engineering-honesty layer that no comparable open project offers:

- **Real try-on rendering**, not overlays â€” your pose, background, face and tattoo are re-rendered by a 972M-parameter MMDiT diffusion model conditioned on the actual garment image
- **Fit Preference Comparison** â€” Slim / Regular / Oversized previews from one render (documented, anchored, non-rigid warp â€” not a resize)
- **Measurement-based fit analysis** â€” *your* chest/waist/shoulder vs the garment's flat measurements, through transparent, editable sizing rules (never invented from the photo)
- **Garment Detail Preservation checker** â€” measured colour Î”E / chroma histograms, ORB feature matching, CLIP embeddings, OCR token checks â€” compared against the *source* garment
- **Try-On Reliability report** â€” measured (face-region SSIM, pose-landmark drift) vs qualitative, with **no fabricated confidence percentages**
- **Local-first privacy** â€” uploads never leave your machine; history only persists if you explicitly save it

This is a working product, not a mockup: upload â†’ analyze â†’ render â†’ download, running on a single RTX 4060 Laptop in under a minute.

---

## What it does

| # | Feature | How it works |
|---|---|---|
| 1 | **AI virtual try-on** | FASHN VTON v1.5 (Apache-2.0), maskless diffusion rendering, seeded & reproducible |
| 2 | **Fit preference simulator** | Slim/Regular/Oversized via torso-anchored sine-profile warp of the rendered garment region ([docs/FIT_WARP.md](docs/FIT_WARP.md)) |
| 3 | **Body measurement assistant** | Manual entry + optional MediaPipe pose-based estimation (explicitly labelled a rough estimate) |
| 4 | **Garment measurement analysis** | Configurable ease/delta rules vs your chest/waist/hip/shoulder â€” transparent & shown in the UI |
| 5 | **Garment detail preservation** | Lab chroma histogram + CIE76 Î”E, HSV correlation, edge-density ratio, ORB + RANSAC inliers, CLIP ViT-B/32 cosine, EasyOCR token recall |
| 6 | **Reliability report** | Face-detection-aligned SSIM, pose-landmark drift, measurement availability, honest qualitative uncertainties |
| 7 | **Outfit recommendations** | Rule-based complementary/analogous colour pairing from the garment's dominant palette |

Every number shown to the user is either (a) **measured** from the two images involved, or (b) **explicitly labelled qualitative**. There are no invented "87% match" scores anywhere in the app.

---

## Methodology

### 1. The generative core â€” FASHN VTON v1.5 (Apache-2.0)

We evaluated CatVTON, IDM-VTON and FASHN VTON (see [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md)). We chose **FASHN VTON v1.5** because:

| Criterion | CatVTON | IDM-VTON | **FASHN VTON v1.5** |
|---|---|---|---|
| License | CC BY-NC-SA 4.0 (non-commercial) | CC BY-NC-SA 4.0 | **Apache-2.0** |
| Parameters | ~900M | ~1.1B + SDXL | **972M** |
| VRAM | <8 GB | 8â€“12 GB | **~8 GB (bf16, measured 3.9â€“7.9 GB)** |
| Architecture | SD-1.5 inpainting + concatenative | IDM (two-stage) | **MMDiT (FLUX-style), pixel-space** |

**Why pixel-space matters.** Latent-diffusion VITON pipelines re-encode through a VAE, so fine prints, logos and textures are lossy before the model ever sees them. FASHN VTON operates directly on RGB pixels (12Ã—12 patch embedding), which is why chest-logo text survives in our renders â€” see the preservation numbers below.

**How the model works, end to end:**

1. **Person preprocessing.** The photo is aspect-preserved-resized to a square canvas, then padded to the model's 576Ã—864. DWPose extracts whole-body keypoints; the person is rendered to a pose image (grayscale, 18 body points + hands/face).
2. **Garment preprocessing.** The garment photo is segmented with the FASHN human parser; for flat-lay garments the parser is skipped and a dummy pose is used (exactly as the model was trained). Pixels not on the garment are set to mid-gray.
3. **Clothing-agnostic person.** The upper-body garment region of the *person* image is masked out (neural parser â†’ morphology), producing the model's "agnostic" input â€” the model never sees the original shirt it must replace.
4. **Diffusion sampling.** A rectified-flow MMDiT (8 double-stream cross-modal blocks + 16 single-stream self-attention blocks, hidden 1280, 10 heads, RoPE positional embeddings, AdaLN modulation conditioned on timestep and garment category embedding) generates the image in `num_timesteps` steps with an EDM-style shifted schedule (`mu=1.5`) and classifier-free guidance (`guidance_scale`, default 1.5, CFG skipped on the last step to avoid colour saturation).

```python
# What actually happens inside the vendor pipeline (simplified)
timesteps = get_rf_schedule(num_steps=20, mu=1.5)   # 0â†’1 shifted schedule
images = torch.randn_like(x0)                        # start from pure noise
for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    v = model.forward_for_cfg(images, t, ca_images, garment_images, poses, ...)
    v_guided = v_u + 1.5 * (v_c - v_u)               # CFG (skipped last step)
    images = images + (t_prev - t_curr) * v_guided   # Euler integration
```

**Two render modes are supported:**
- **Maskless (default)** â€” the whole frame is re-rendered; best face/garment/background coherence (this is what FASHN recommends and trains primarily for)
- **Masked** â€” non-garment pixels are kept 1:1 from the original; explicit choice when the original outfit silhouette differs greatly from the new one (`segmentation_free=false` in the API / the "Render mode" toggle in the UI)

**Output contract:** 576Ã—864 RGB PNG, deterministic given `seed` â€” every render in the UI shows its steps and seed and exports are byte-reproducible.

### 2. Input understanding â€” MediaPipe Pose + quality gates

Before any GPU work happens, the backend validates both images:

- **Person:** MediaPipe Pose (model complexity 1) detects shoulders, elbows, wrists, hips, knees, ankles. We gate on (a) resolution â‰¥360 px short side, (b) variance-of-Laplacian sharpness â‰¥60, (c) brightness 30â€“235. Detected pose also powers the optional measurement estimator (scaled strictly from user-provided height, with explicit "rough estimate only" labelling â€” never claimed to be accurate).
- **Garment:** rembg (uÂ²-Net) removes the background (CPU, transparent checkerboard in the studio preview), dominant colours are extracted with `cv2.kmeans` (k=4, share-sorted), ORB keypoint density is reported, and resolution/sharpness gates mirror the person checks.
- **Garment category** (`tops` | `bottoms`) selects the model's mask labels and the fit-warp band.

### 3. Fit-preference simulation â€” documented approximation, not fake physics

The try-on model has **no "fit" control**. Instead, after rendering we apply an **anatomy-anchored, non-uniform warp** to the garment region:

```
for each row y in the garment band:
    w(y) = sin(Ï€ Â· t(y))^0.85        # 0 at band edges â†’ 1 at mid-torso
    s(y) = 1 + (target âˆ’ 1) Â· w(y)   # Â±8â€“9% horizontal scale at mid-torso
    x'   = cx + (x âˆ’ cx) / s(y)      # remap around torso axis
```

Band edges are feather-blended (alpha ramps over ~12% of band height) so the head, background and untouched legs blend seamlessly. Gains are pixels-only â€” the garment's print, logo and weave are **resampled, never redrawn**.

| Fit | Mid-torso horizontal scale | Vertical stretch |
|---|---|---|
| Slim | Ã—0.92 | Ã—0.99 |
| Regular | Ã—1.00 (identity) | Ã—1.00 |
| Oversized | Ã—1.09 | Ã—1.03 |

This is explicitly labelled a **visual approximation**, and the *quantitative* fit verdict (Section 5) never reads the pixels â€” it only reads your measurements. See [docs/FIT_WARP.md](docs/FIT_WARP.md).

### 4. Garment Detail Preservation checker

We compare the **uploaded garment** against the **garment region of the render** (auto-cropped from pose landmarks; falls back to the central third if pose fails):

1. **Colour similarity** â€” Lab (a,b) chroma histogram correlation (lighting-invariant) + raw HSV histogram correlation (reported, not gated) + dominant-colour Î”E with greedy nearest-cluster matching. Î±-channel-respecting via rembg cutout on the garment side.
2. **Pattern preservation** â€” Canny edge-density ratio (r/g should be â‰ˆ0.5â€“2.2) + ORB feature matches with  ratio <0.78 (optionally homography with RANSAC inliers when â‰¥8 good matches). A flat-lay vs worn garment *never* matches pixel-for-pixel; sparse matches are expected and reported honestly.
3. **Semantic appearance** â€” CLIP ViT-B/32 image embeddings (transformers, CPU): cosine similarity of the white-composited garment cutout vs the rendered region.
4. **Logo / text** â€” EasyOCR (en) tokenized comparison with substring-tolerant recall (â‰¥0.6 passes; OCR on garments is noisy by nature).

**Result classification:** `preserved` / `potentially_altered` / `not_verified`. The first two are *measured*; the third is used whenever the pipeline cannot honestly verify a claim (e.g. buttons/seams below automated resolution). Nothing is ever auto-marked "perfect".

### 5. Measurement-based fit analysis â€” transparent rules, shown to the user

From your measured chest/waist/hip/shoulder and the garment's flat spec, we compute:

- **Chest ease** = 2 Ã— chest width âˆ’ your chest â†’ verdict `tight / ideal / relaxed / loose` vs the ease band for the selected fit (Slim 2â€“9 cm, Regular 9â€“16 cm, Oversized 16â€“30 cm)
- **Waist / hip** ease with analogous bands
- **Shoulder delta** vs target delta range (Slim âˆ’3â€¦+1, Regular 0â€¦+3, Oversized +2â€¦+6)
- **Length** vs a 0.30 Ã— height torso-length estimate
- **Overall** and recommended fit derived from the per-area verdicts (â‰¥2 tight â†’ size up suggestion; â‰¥2 loose â†’ size down)

Every constant lives in `backend/app/services/fit_analysis.py::FIT_RULES` and is echoed back to the UI as "Rules used" â€” the user audits exactly what "ideal" means. Nothing is invented from the garment photo; missing inputs are reported as "not enough measurements", not filled with guesses.

### 6. Reliability report

A per-generation checklist distinguishing **measured** from **qualitative**:

| Check | Type | Method |
|---|---|---|
| Person identity consistency | measured | Face crops detected via MediaPipe FaceDetection, normalized to 128Ã—128, SSIM |
| Garment colour | measured | Preservation metrics (5.1) |
| Pattern | measured | Preservation metrics (5.2) |
| Logo/text | measured / unverified | OCR recall (EasyOCR) |
| Body-region alignment | measured | MediaPipe pose landmark drift of shoulder/hip axis between input & render (<8% = stable) |
| Measurement availability | measured | Whether you filled body/garment forms |
| Areas of uncertainty | qualitative | Honest list (drape, hands near garments, longâ†’short swaps, lighting) |

No confidence percentages are emitted anywhere, because diffusion sampling has no calibrated probability to report.

### 7. Outfit recommendations

The garment's dominant palette (k-means, share-sorted) is mapped through HSV hue analysis to 2â€“4 complementary/analogous/neutral accent swatches, plus rule-based trousers/shoes/accessory guidance per category. Explicitly framed as "colour-theory starting points, not fashion dictation".

---

## Architecture

<p align="center">
  <img src="docs/assets/fig_architecture.svg" alt="FitVision AI inference pipeline" width="100%">
</p>

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  person.jpg â”‚â”€â”€â”  â”‚  FastAPI job runner (single-GPU lock)        â”‚  â”Œâ”€â–¶â”‚ result PNG     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”œâ”€â–¶â”‚   1. pose & quality gates (MediaPipe)         â”‚  â”‚  + 3 fit warps â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚  â”‚   2. garment cutout & palette (rembg+kmeans) â”‚  â”‚                â”‚
â”‚ garment.jpg â”‚â”€â”€â”˜  â”‚   3. FASHN VTON v1.5 render (bf16, seeded)   â”‚â”€â”€â”¤                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â”‚   4. warp slim/regular/oversized previews     â”‚  â”‚  preservation & â”‚
                    â”‚   5. preservation + reliability analyses      â”‚  â”‚  reliability    â”‚
                    â”‚   6. fit analysis from your measurements      â”‚  â”‚  reports        â”‚
                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Stack:** React 19 + Vite + Tailwind v4 (frontend) Â· FastAPI + SQLAlchemy + SQLite (backend) Â· PyTorch 2.6 + CUDA 12.4 (model) Â· OpenCV / MediaPipe / EasyOCR / CLIP / rembg / Pillow (CV) Â· no paid API anywhere.

**Privacy:** uploads live only for the duration of the job (`backend/data/uploads/<job>` deleted post-render); results are held ephemerally for 24 h and deleted on server restart unless you tick "Save to history" â€” only then are person/garment copies retained in `backend/data/results/<job>` alongside a SQLite metadata row.

---

## Performance

Measured on the reference machine (Intel i5 Â· 16 GB RAM Â· **RTX 4060 Laptop 8 GB** Â· 576Ã—864 output):

<p align="center">
  <img src="docs/assets/fig_benchmark.png" alt="Measured wall-clock per preset" width="78%">
</p>

<p align="center">
  <img src="docs/assets/fig_preservation.png" alt="Preservation metrics by preset" width="78%">
</p>

| Metric | Draft (4 steps) | Fast (8) | Balanced (12) | Best (20) |
|---|---|---|---|---|
| Diffusion sampling | ~6 s | ~13 s | ~18 s | ~32 s |
| End-to-end (clickâ†’result) | ~13 s | ~24 s | ~31 s | ~46 s |
| CLIP cosine vs source garment | 0.838 | 0.862 | 0.855 | **0.868** |
| Chroma histogram corr. (Lab a/b) | 0.94 | 0.94 | 0.93 | 0.94 |
| Logo OCR recall (FASHN AI chest print) | 1.0 | 1.0 | 1.0 | 1.0 |
| GPU memory (peak) | ~7.9 GB | ~7.9 GB | ~7.9 GB | ~7.9 GB |

First render after a cold boot pays ~40 s one-time model load (pre-warmed at startup in the background â€” subsequent renders skip it). Only one render runs at a time (server-side lock; concurrent requests receive HTTP 409 rather than silently sharing the GPU).

---

## Quick start

### Prerequisites
- Windows with NVIDIA GPU â‰¥ 6 GB VRAM (tested: RTX 4060 Laptop 8 GB) Â· Python 3.12 Â· Node â‰¥ 20

### 1 Â· Backend
```powershell
cd backend
python -m pip install -r requirements.txt
cd ..\ai\fashn-vton-1.5
python -m pip install -e .
python scripts/download_weights.py --weights-dir ./weights   # ~2 GB, one time
cd ..\..
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8017
```
> Always start with `python -m uvicorn`, never a bare `uvicorn` (on Windows machines with multiple Pythons installed, the bare CLI may resolve to the wrong interpreter â€” this bit us hard on this project).

### 2 Â· Frontend
```powershell
cd frontend
npm install
npm run build        # production build (served by `npm run preview`)
npm run preview -- --port 5173
```
Open **http://127.0.0.1:5173** â†’ *Try-On Studio*. The "Try with sample photos" button needs zero uploads â€” bundled sample images are included.

### 3 Â· One-click launcher
```powershell
powershell -ExecutionPolicy Bypass -File docs\start.ps1
```

### 4 Â· Try it via the API (no UI)
```bash
curl -X POST http://127.0.0.1:8017/api/tryon/generate \
  -F person=@person.jpg -F garment=@garment.jpg \
  -F category=tops -F fit=regular -F steps=20 -F seed=42
```

---

## Repository layout

```
â”œâ”€â”€ ai/fashn-vton-1.5/     # submodule â€” FASHN VTON v1.5 inference code (Apache-2.0) + weights*
â”œâ”€â”€ backend/               # FastAPI application
â”‚   â”œâ”€â”€ app/routers/       # /api/health Â· /api/analyze/* Â· /api/tryon/* Â· /api/profile Â· /api/history
â”‚   â”œâ”€â”€ app/services/      # tryon_service Â· pose Â· garment Â· preservation Â· reliability
â”‚   â”‚                      # fit_analysis Â· fit_warp Â· recommendations Â· image_utils
â”‚   â”œâ”€â”€ tests/             # pytest â€” 8 passing (fit rules, warp, recommendations)
â”‚   â”œâ”€â”€ training/          # LoRA fine-tuning pipeline (see docs/TRAINING.md) â€” unused at runtime
â”‚   â””â”€â”€ data/              # SQLite + ephemeral/saved results (git-ignored)
â”œâ”€â”€ frontend/              # React 19 + Vite + Tailwind v4
â”‚   â””â”€â”€ src/{pages,components,lib}
â”œâ”€â”€ docs/                  # README-adjacent docs: FIT_WARP Â· MODEL_SELECTION Â· TRAINING
â”‚                          # OWN_MODEL Â· PRIVACY Â· TESTING Â· start.ps1 Â· assets/
â””â”€â”€ README.md              # this file
```

*Weights (â‰ˆ2 GB) are git-ignored; fetch once with `ai/fashn-vton-1.5/scripts/download_weights.py`.
To use **your own fine-tuned checkpoint**, point `FASHN_WEIGHTS` at its directory â€” see [docs/OWN_MODEL.md](docs/OWN_MODEL.md).

## Testing

```powershell
cd backend
python -m pytest tests/ -q      # 8 passing â€” fit rules, warp geometry, recommendations
```
Manual & E2E checklists: [docs/TESTING.md](docs/TESTING.md) (invalid uploads, missing garment, GPU OOM, history save/delete, 24 h TTL cleanup, layout checks).

## API overview

| Method | Route | Purpose |
|---|---|---|
| GET  | `/api/health` | Engine status (availability, label, weights size, loaded) |
| POST | `/api/analyze/person` | Pose detection, quality gates, optional height-scaled estimates |
| POST | `/api/analyze/garment` | Cutout, dominant colours, warnings |
| POST | `/api/fit-analysis` | Standalone measurement â†’ verdicts |
| POST | `/api/tryon/generate` | Queue a try-on job (multipart) |
| GET  | `/api/jobs/{id}` | Poll status / result |
| GET  | `/api/results/{id}/image/{slim\|regular\|oversized}` | Rendered images |
| GET/PUT | `/api/profile` | Local body-measurement profile (SQLite) |
| GET/DELETE | `/api/history` Â· `/api/history/{id}` | Saved history CRUD |

Full interactive docs at `http://127.0.0.1:8017/docs`.

## Honest limitations

- Diffusion renders are **plausible images, not measurements**; 576Ã—864 is the model's fixed resolution; several minutes are the cost of real local inference on consumer GPUs.
- Fit previews are **visual approximations** (documented warp), not cloth simulation; physical fit cannot be derived from a 2D render â€” the measurement-based verdict is the only quantitative fit statement.
- Photo-based measurement estimates need a full-body frame + your real height; they are rough (camera, lens, clothing, pose all distort scale) and never treated as ground truth.
- Hands/fingers are the weakest region for any diffusion model; if a render breaks them, re-roll the seed or try Masked mode (keeps non-garment pixels 1:1).
- VITON-HD (the dataset used to explore fine-tuning) is **CC BY-NC-SA 4.0** â€” research only; the app does **not** ship VITON-HD-derived weights.
- Preservation checks are decision-support indicators, not proof of design fidelity; flat-lay vs worn prints will never feature-match strongly; trust CLIP + visual inspection.
- Body alignment drift and face SSIM are diagnostics, not biometric verification.

## License & credits

- This repository: **MIT**.
- [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) â€” **Apache-2.0** â€” Â© FASHN AI
- MediaPipe Pose â€” Apache-2.0 Â· rembg / uÂ²-Net â€” MIT/Apache-2.0 Â· EasyOCR â€” Apache-2.0 Â· OpenAI CLIP weights â€” MIT (via HuggingFace `transformers`) Â· OpenCV â€” Apache-2.0
- No paid API is used anywhere in this project. Model weights are downloaded once from HuggingFace.

## Roadmap

- [ ] Stream try-on images progressively (deliver render before analysis finishes; show reports as they land)
- [ ] ONNX-Runtime-GPU garment analysis (rembg currently falls back to CPU)
- [ ] Lower-body (`bottoms`) quality hardening & leg-slot fit warp bands
- [ ] Multi-garment outfit stacking (top + bottom in one session)
- [ ] Fine-tuned in-domain LoRA (Apache-2.0 data only) once evaluated against stock
