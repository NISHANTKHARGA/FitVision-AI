"""Core API routes: analysis, generation, jobs, results, fit analysis."""
import json
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import (UPLOAD_DIR, RESULTS_DIR, MAX_UPLOAD_MB, ALLOWED_EXTENSIONS,
                      FASHN_AVAILABLE, DEFAULT_STEPS, MAX_STEPS)
from ..db import SessionLocal
from ..models_db import ResultRecord
from ..services.garment import analyze_garment
from ..services.image_utils import load_image
from ..services.pose import analyze_person
from ..services.tryon_service import (create_job, get_job, run_generation_job,
                                      fashn_status, _update, _generation_lock)
from ..services.fit_analysis import analyze_fit
from ..schemas import BodyMeasurements, GarmentMeasurements

router = APIRouter(prefix="/api", tags=["tryon"])


def _validate_upload(file: UploadFile) -> str:
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    return ext


async def _read_limited(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    if len(data) < 1024:
        raise HTTPException(400, "File is too small to be a valid image.")
    return data


@router.get("/health")
def health():
    return {
        "status": "ok",
        "fashn": fashn_status(),
        "default_steps": DEFAULT_STEPS,
    }


@router.post("/analyze/person")
async def analyze_person_route(file: UploadFile = File(...), height_cm: float | None = Form(None)):
    _validate_upload(file)
    data = await _read_limited(file)
    try:
        rgb = load_image(data)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode image: {exc}")
    return analyze_person(rgb, height_cm=height_cm)


@router.post("/analyze/garment")
async def analyze_garment_route(file: UploadFile = File(...)):
    _validate_upload(file)
    data = await _read_limited(file)
    try:
        rgb = load_image(data)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode image: {exc}")
    return analyze_garment(rgb)


@router.post("/fit-analysis")
def fit_analysis_route(payload: dict):
    body = payload.get("body") or {}
    garment = payload.get("garment") or {}
    category = payload.get("category", "tops")
    fit = payload.get("fit", "regular")
    if category not in ("tops", "bottoms"):
        raise HTTPException(400, "category must be 'tops' or 'bottoms'")
    # Validate numeric fields through pydantic (rejects absurd values)
    BodyMeasurements(**body)
    GarmentMeasurements(**garment)
    return analyze_fit(body, garment, category, fit)


@router.post("/tryon/generate")
async def generate_route(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    category: str = Form(...),
    fit: str = Form("regular"),
    steps: int = Form(DEFAULT_STEPS),
    seed: int = Form(42),
    garment_photo_type: str = Form("model"),
    guidance_scale: float = Form(1.5),
    segmentation_free: bool = Form(True),
    save_history: bool = Form(False),
    body_measurements: str = Form("{}"),
    garment_measurements: str = Form("{}"),
):
    if not FASHN_AVAILABLE:
        raise HTTPException(503, "FASHN VTON weights are not available on this machine. See README for setup.")
    if _generation_lock.locked():
        raise HTTPException(409, "A try-on is currently rendering on this machine. Wait for it to finish and try again.")
    if category not in ("tops", "bottoms"):
        raise HTTPException(400, "category must be 'tops' or 'bottoms'")
    if fit not in ("slim", "regular", "oversized"):
        raise HTTPException(400, "fit must be slim | regular | oversized")
    if garment_photo_type not in ("model", "flat-lay"):
        raise HTTPException(400, "garment_photo_type must be 'model' or 'flat-lay'")
    steps = max(4, min(MAX_STEPS, int(steps)))
    guidance_scale = max(1.0, min(3.0, float(guidance_scale)))

    try:
        body_m = BodyMeasurements(**(json.loads(body_measurements or "{}"))).model_dump(exclude_none=True)
    except Exception as exc:
        raise HTTPException(400, f"body_measurements invalid: {exc}")
    try:
        garment_m = GarmentMeasurements(**(json.loads(garment_measurements or "{}"))).model_dump(exclude_none=True)
    except Exception as exc:
        raise HTTPException(400, f"garment_measurements invalid: {exc}")

    _validate_upload(person)
    _validate_upload(garment)
    person_bytes = await _read_limited(person)
    garment_bytes = await _read_limited(garment)

    # Decode check before queueing
    try:
        load_image(person_bytes)
        load_image(garment_bytes)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode one of the images: {exc}")

    job_id = create_job()
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    person_path = job_dir / ("person" + Path(person.filename or "person.png").suffix.lower())
    garment_path = job_dir / ("garment" + Path(garment.filename or "garment.png").suffix.lower())
    person_path.write_bytes(person_bytes)
    garment_path.write_bytes(garment_bytes)

    if segmentation_free not in (True, False):
        raise HTTPException(400, "segmentation_free must be true or false")

    def worker():
        try:
            run_generation_job(job_id, person_path, garment_path, category, fit,
                               steps, seed, garment_photo_type, save_history,
                               body_m, garment_m, guidance_scale, segmentation_free)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            # Record metadata (always metadata-only unless user opted in to keep images)
            try:
                job = get_job(job_id)
                if job and job.get("status") == "done":
                    with SessionLocal() as db:
                        db.merge(ResultRecord(
                            id=job_id, category=category, fit=fit, steps=steps, seed=seed,
                            saved=save_history,
                            has_measurements=bool(body_m or garment_m),
                            summary_json=json.dumps({
                                "overall": (job.get("result") or {}).get("fit_analysis", {}).get("overall", ""),
                                "preservation_summary": (job.get("result") or {}).get("preservation", {}).get("summary", ""),
                            }),
                            image_path=f"{job_id}/tryon_{fit}.png" if save_history else None,
                            person_image_path=f"{job_id}/person_input.png" if save_history else None,
                        ))
                        db.commit()
                elif job and job.get("status") == "error":
                    with SessionLocal() as db:
                        db.merge(ResultRecord(id=job_id, category=category, fit=fit,
                                              steps=steps, seed=seed, saved=False,
                                              summary_json=json.dumps({"error": job.get("error")})))
                        db.commit()
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        # Fall back to DB (job from a previous session)
        with SessionLocal() as db:
            rec = db.get(ResultRecord, job_id)
        if rec:
            return {"job_id": job_id,
                    "status": "done" if rec.saved else "expired",
                    "stage": "done" if rec.saved else "expired",
                    "message": "Result available in history." if rec.saved else
                               "This ephemeral result is no longer available. Generate again, or save results to history.",
                    "progress": 1.0, "result": None, "error": None,
                    "saved_in_db": bool(rec.saved)}
        raise HTTPException(404, "Unknown job id")
    return job


@router.get("/results/{job_id}/image/{variant}")
def result_image(job_id: str, variant: str):
    if variant not in ("slim", "regular", "oversized"):
        raise HTTPException(400, "variant must be slim | regular | oversized")
    if "/" in job_id or ".." in job_id:
        raise HTTPException(400, "invalid id")
    path = RESULTS_DIR / job_id / f"tryon_{variant}.png"
    if not path.exists():
        raise HTTPException(404, "Image not found (unsaved results are kept for a limited time).")
    return FileResponse(path, media_type="image/png",
                        headers={"Content-Disposition": f'inline; filename="fitvision_{variant}.png"'})


@router.get("/results/{job_id}/source/{kind}")
def result_source(job_id: str, kind: str):
    if kind not in ("person", "garment"):
        raise HTTPException(400, "kind must be person | garment")
    path = RESULTS_DIR / job_id / f"{kind}_input.png"
    if not path.exists():
        raise HTTPException(404, "Source image was not retained (only saved history keeps uploads).")
    return FileResponse(path, media_type="image/png")


@router.post("/history/save/{job_id}")
def save_to_history(job_id: str):
    """Persist an ephemeral result to history (images become kept permanently)."""
    with SessionLocal() as db:
        rec = db.get(ResultRecord, job_id)
        if not rec:
            raise HTTPException(404, "Unknown result id")
        rec.saved = True
        rec.image_path = f"{job_id}/tryon_{rec.fit}.png"
        db.commit()
    return {"saved": True}
