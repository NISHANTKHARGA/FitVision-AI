#Final connection
"""FASHN VTON v1.5 pipeline wrapper + asynchronous job manager.

The model is loaded lazily on first use and kept in memory afterwards.
Generation is slow on consumer GPUs (minutes, not seconds), so jobs run on a
background thread and the frontend polls /api/jobs/{id}.
"""
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import FASHN_REPO, FASHN_WEIGHTS, FASHN_AVAILABLE, RESULTS_DIR, MODEL_LABEL

_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_error = None


def fashn_status() -> dict:
    return {
        "available": FASHN_AVAILABLE,
        "weights_dir": str(FASHN_WEIGHTS),
        "repo": str(FASHN_REPO),
        "loaded": _pipeline is not None,
        "load_error": _pipeline_error,
        "label": MODEL_LABEL,
        "weights_size_mb": round((FASHN_WEIGHTS / "model.safetensors").stat().st_size / 1e6, 1)
        if (FASHN_WEIGHTS / "model.safetensors").exists() else None,
        "note": "FASHN VTON v1.5 (Apache-2.0). Maskless, pixel-space try-on; ~8 GB VRAM, bfloat16 on RTX 30/40xx. Point FASHN_WEIGHTS at your own checkpoint to use your trained model.",
    }


def get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            import sys
            if str(FASHN_REPO / "src") not in sys.path:
                sys.path.insert(0, str(FASHN_REPO / "src"))
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            from fashn_vton import TryOnPipeline
            _pipeline = TryOnPipeline(weights_dir=str(FASHN_WEIGHTS))
            _pipeline_error = None
            return _pipeline
        except Exception as exc:
            _pipeline_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            raise


# ---------------------------------------------------------------- job manager
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_generation_lock = threading.Lock()   # one GPU render at a time


def create_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id, "status": "queued", "stage": "queued",
            "message": "Job queued…", "progress": 0.0, "result": None, "error": None,
            "created_at": time.time(),
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def run_generation_job(job_id: str, person_path: Path, garment_path: Path,
                       category: str, fit: str, steps: int, seed: int,
                       garment_photo_type: str, save_history: bool,
                       body_measurements: dict, garment_measurements: dict,
                       guidance_scale: float = 1.5, segmentation_free: bool = True):
    """Full pipeline: try-on generation (long) + analyses. Runs on a worker thread."""
    if not _generation_lock.acquire(blocking=False):
        return _fail_busy(job_id)
    try:
        _run_generation_inner(job_id, person_path, garment_path, category, fit, steps, seed,
                              garment_photo_type, save_history, body_measurements, garment_measurements,
                              guidance_scale, segmentation_free)
    finally:
        _generation_lock.release()


def _fail_busy(job_id: str):
    _update(job_id, status="error", stage="error",
            message="Another try-on is already rendering on this machine. Wait for it to finish and retry.",
            progress=1.0, error="GPU busy: only one generation can run at a time.")
    return None


def _run_generation_inner(job_id: str, person_path: Path, garment_path: Path,
                       category: str, fit: str, steps: int, seed: int,
                       garment_photo_type: str, save_history: bool,
                       body_measurements: dict, garment_measurements: dict,
                       guidance_scale: float = 1.5, segmentation_free: bool = True):
    """Worker body (generation lock held by caller)."""
    from .pose import pose_keypoints, quality_check
    from .garment import analyze_garment, cutout
    from .fit_warp import fit_warp
    from .preservation import check_preservation
    from .reliability import build_report
    from .fit_analysis import analyze_fit
    from .recommendations import recommend
    from .image_utils import load_image

    try:
        _update(job_id, status="running", stage="loading",
                message="Loading the try-on model (first run downloads the Xet cache and may take a minute)…",
                progress=0.05)
        pipeline = get_pipeline()

        person_rgb = load_image(person_path)
        garment_rgb = load_image(garment_path)

        _update(job_id, stage="preprocess",
                message="Analysing person and garment (pose, colours, background removal)…",
                progress=0.10)
        pose_person = pose_keypoints(person_rgb)

        _update(job_id, stage="generating",
                message=f"Generating try-on — {steps} steps, guidance {guidance_scale}…",
                progress=0.15)
        result = pipeline(
            person_image=Image.fromarray(person_rgb),
            garment_image=Image.fromarray(garment_rgb),
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            segmentation_free=bool(segmentation_free),
        )
        out_img = result.images[0].convert("RGB")
        out_rgb = np.array(out_img)

        out_dir = RESULTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        main_path = out_dir / "tryon_regular.png"
        out_img.save(main_path)

        # ------- Fit-preference previews (documented geometric approximation) -------
        _update(job_id, stage="fit-variants",
                message="Rendering slim / regular / oversized fit previews (torso-anchored warp, see docs)…",
                progress=0.80)
        pose_result = pose_keypoints(out_rgb) or pose_person
        variant_paths = {}
        for f in ("slim", "regular", "oversized"):
            warped = fit_warp(out_rgb, pose_result, category, f)
            p = out_dir / f"tryon_{f}.png"
            Image.fromarray(warped).save(p)
            variant_paths[f] = str(p)

        # Keep uploads alongside the ephemeral result so the before/after view works.
        # Unsaved results auto-expire (see cleanup on startup); saved history persists them.
        person_keep = out_dir / "person_input.png"
        garment_keep = out_dir / "garment_input.png"
        Image.fromarray(person_rgb).save(person_keep)
        Image.fromarray(garment_rgb).save(garment_keep)

        # ------- Analyses (garment analysis and preservation run in parallel) -------
        _update(job_id, stage="analysis",
                message="Checking garment preservation (colour, pattern, CLIP, OCR)…",
                progress=0.88)
        from concurrent.futures import ThreadPoolExecutor
        cutout_parts = None
        gmask = None
        try:
            cutout_parts = cutout(garment_rgb)   # garment on transparent bg — used by preview metrics & preservation
            gmask = cutout_parts[1]
        except Exception:
            pass
        with ThreadPoolExecutor(max_workers=2) as pool:
            garment_future = pool.submit(analyze_garment, garment_rgb, cutout_parts)
            preservation_future = pool.submit(check_preservation, garment_rgb, gmask, out_rgb, pose_result, category)
            garment_analysis = garment_future.result()
            preservation = preservation_future.result()
        person_quality = quality_check(person_rgb)
        reliability = build_report(
            person_rgb, out_rgb, pose_person, pose_result, preservation,
            {"body": body_measurements, "garment": garment_measurements},
            person_quality,
        )
        fit_report = analyze_fit(body_measurements or {}, garment_measurements or {}, category, fit)
        recs = recommend(garment_analysis.get("dominant_colors", []), category)

        result_dict = {
            "job_id": job_id,
            "category": category,
            "fit": fit,
            "garment_photo_type": garment_photo_type,
            "steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "segmentation_free": bool(segmentation_free),
            "images": {f: f"/api/results/{job_id}/image/{f}" for f in variant_paths},
            "main_image": f"/api/results/{job_id}/image/{fit}",
            "fit_variants_note": ("Slim/Regular/Oversized previews are produced by a documented torso-anchored "
                                  "non-rigid warp of the generated garment region — a visual approximation, "
                                  "not a physical fit simulation. See docs/FIT_WARP.md."),
            "preservation": preservation,
            "reliability": reliability,
            "fit_analysis": fit_report,
            "recommendations": recs,
            "garment": {
                "dominant_colors": garment_analysis.get("dominant_colors", []),
                "background_removed": garment_analysis.get("background_removed"),
                "warnings": garment_analysis.get("warnings", []),
            },
            "saved_to_history": bool(save_history),
            "attribution": "Try-on generated locally with FASHN VTON v1.5 (Apache-2.0).",
        }

        _update(job_id, status="done", stage="done", message="Done.",
                progress=1.0, result=result_dict)
        return result_dict

    except Exception as exc:
        traceback.print_exc()
        _update(job_id, status="error", stage="error",
                message="Generation failed.",
                progress=1.0,
                error=f"{type(exc).__name__}: {exc}")
        return None
