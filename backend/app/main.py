"""FitVision AI — FastAPI application entry point."""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import RESULTS_DIR, EPHEMERAL_RESULT_TTL_HOURS
from .db import init_db, SessionLocal
from .models_db import ResultRecord
from .routers import tryon, profile


def cleanup_ephemeral_results():
    """Delete unsaved result folders older than the TTL. Saved history is kept."""
    now = time.time()
    ttl = EPHEMERAL_RESULT_TTL_HOURS * 3600
    with SessionLocal() as db:
        saved_ids = {r.id for r in db.query(ResultRecord.id).filter(ResultRecord.saved == True).all()}  # noqa: E712
    for child in RESULTS_DIR.iterdir():
        if not child.is_dir():
            continue
        if child.name in saved_ids:
            continue
        try:
            if now - child.stat().st_mtime > ttl:
                import shutil
                shutil.rmtree(child, ignore_errors=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    cleanup_ephemeral_results()
    # Pre-warm the try-on model in the background so the first user generation
    # doesn't pay the ~40 s model-load penalty.
    from .services.tryon_service import fashn_status
    import threading
    if fashn_status()["available"]:
        threading.Thread(target=_prewarm, daemon=True).start()
    yield


def _prewarm():
    import logging
    from .services.tryon_service import get_pipeline
    try:
        get_pipeline()
        logging.info("FASHN VTON pipeline pre-warmed.")
    except Exception as exc:
        logging.warning("Pipeline pre-warm failed: %s", exc)


app = FastAPI(title="FitVision AI", version="1.0.0",
              description="Intelligent 2D virtual try-on with fit simulation, garment preservation analysis and measurement-based fit insights.",
              lifespan=lifespan)

import os
_cors_origins = os.environ.get(
    "FITVISION_CORS_ORIGINS",
    "https://fitvision-ai-cyan.vercel.app",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tryon.router)
app.include_router(profile.router)


@app.get("/")
def root():
    return {"app": "FitVision AI", "docs": "/docs", "health": "/api/health"}
