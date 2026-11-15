# ==========================================
# Try-On Router
# Handles virtual try-on API endpoints.
# ==========================================

"""Profile + history routes."""
import json
import shutil

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import RESULTS_DIR
from ..db import SessionLocal
from ..models_db import BodyProfile, ResultRecord
from ..schemas import BodyMeasurements

router = APIRouter(prefix="/api", tags=["profile"])


class ProfileUpdate(BodyMeasurements):
    pass


@router.get("/profile")
def get_profile():
    with SessionLocal() as db:
        p = db.get(BodyProfile, 1)
        if not p:
            return {"height_cm": None, "chest_cm": None, "waist_cm": None,
                    "shoulder_cm": None, "hip_cm": None, "preferred_fit": "regular"}
        return {"height_cm": p.height_cm, "chest_cm": p.chest_cm, "waist_cm": p.waist_cm,
                "shoulder_cm": p.shoulder_cm, "hip_cm": p.hip_cm,
                "preferred_fit": p.preferred_fit or "regular"}


@router.put("/profile")
def put_profile(payload: ProfileUpdate):
    with SessionLocal() as db:
        p = db.get(BodyProfile, 1)
        if not p:
            p = BodyProfile(id=1)
            db.add(p)
        for field, value in payload.model_dump().items():
            setattr(p, field, value)
        db.commit()
    return {"saved": True}


@router.get("/history")
def get_history():
    with SessionLocal() as db:
        rows = db.query(ResultRecord).filter(ResultRecord.saved == True) \
                 .order_by(ResultRecord.created_at.desc()).all()  # noqa: E712
        out = []
        for r in rows:
            try:
                summary = json.loads(r.summary_json or "{}")
            except Exception:
                summary = {}
            out.append({
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "category": r.category,
                "fit": r.fit,
                "steps": r.steps,
                "has_measurements": r.has_measurements,
                "image_url": f"/api/results/{r.id}/image/{r.fit}",
                "summary": summary,
            })
        return {"items": out}


@router.delete("/history/{job_id}")
def delete_history(job_id: str):
    if "/" in job_id or ".." in job_id:
        raise HTTPException(400, "invalid id")
    with SessionLocal() as db:
        rec = db.get(ResultRecord, job_id)
        if rec:
            db.delete(rec)
            db.commit()
    shutil.rmtree(RESULTS_DIR / job_id, ignore_errors=True)
    return {"deleted": True}
