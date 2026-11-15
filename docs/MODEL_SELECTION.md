# Model Selection — Evaluation Notes (November 2026)

| Candidate | License | Weights | VRAM | Needs | Verdict |
|---|---|---|---|---|---|
| **FASHN VTON v1.5** (chosen) | **Apache-2.0** | ~2 GB | ~8 GB (bf16, Ampere+) | DWPose ONNX + human parser (auto) | ✅ Best fit. Maskless, pixel-space (preserves prints/logos better than VAE latents), commercial-friendly, simple `pip install -e .` |
| CatVTON | CC BY-NC-SA 4.0 (non-commercial) | ~1 GB | < 8 GB @ 1024×768 | SCHP + DensePose for masks | ❌ Non-commercial license; heavier setup (patched Diffusers fork); superseded v1.5 on quality |
| IDM-VTON | CC BY-NC-SA 4.0 (non-commercial) | ~10+ GB (SDXL) | ~8–12 GB | DensePose + human parsing | ❌ Heavy; non-commercial; slower on 4060 |
| FASHN API | Commercial SaaS | — | — | Paid API | ❌ Requirement: local, free inference |

## Chosen configuration (this machine: RTX 4060 Laptop GPU, 8 GB)

- Precision: **bfloat16** (Ada sm_89 supports it; auto-selected by the pipeline)
- Resolution: 576×864 (model default, fixed)
- Sampling steps: user-selectable speed presets — **Draft 4 g1.0 · Fast 8 g1.2 · Balanced 12 g1.5 · Best 20 g1.5 (default)**
  - Measured end-to-end on this GPU (model load + render + analyses): ~4 min draft, ~5 min fast
  - Only one GPU generation runs at a time; the backend rejects concurrent renders with HTTP 409
- Seed: user-controllable, default 42
- `segmentation_free=True` (maskless) — best body preservation, unconstrained garment volume

## Known fit limitations (also shown in the UI)

- 576×864 output is lower resolution than VAE-based pipelines; set expectations.
- Swapping long→short or bulky→slim garments may leave faint traces of the original garment.
- Faces are re-rendered by the diffusion decoder and can drift slightly; our
  reliability report measures face-region SSIM to catch gross identity breakage.
- Long inference time is a hardware reality on 8 GB laptop GPUs. Draft (8 steps) is usually
  visually very close to 20 steps in our samples. Only one render runs at a time — triggering
  two at once does not speed anything up, it halves throughput for both (enforced server-side).

## Third-party components

FASHN VTON v1.5 (Apache-2.0) · DWPose (Apache-2.0) · YOLOX (Apache-2.0) ·
FASHN Human Parser · rembg/u²net (MIT/Apache-2.0) for garment preview ·
MediaPipe Pose (Apache-2.0) · EasyOCR (Apache-2.0) for logo/text checks ·
OpenAI CLIP ViT-B/32 weights (MIT) via HuggingFace `transformers`.
