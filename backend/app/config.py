"""FitVision AI backend configuration."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
WORKSPACE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "fitvision.db"

# FASHN VTON v1.5 is vendored under ai/fashn-vton-1.5 (Apache-2.0)
FASHN_REPO = WORKSPACE_DIR / "ai" / "fashn-vton-1.5"
FASHN_WEIGHTS = Path(os.environ.get("FASHN_WEIGHTS", FASHN_REPO / "weights"))
FASHN_AVAILABLE = (FASHN_WEIGHTS / "model.safetensors").exists()
# Display label for the loaded checkpoint (set FITVISION_MODEL_LABEL for your own fine-tunes)
MODEL_LABEL = os.environ.get("FITVISION_MODEL_LABEL", "FASHN VTON v1.5 (stock, Apache-2.0)")

# 12 = fast default (~3-4 min on RTX 4060 Laptop), 20 = balanced, 30 = quality
DEFAULT_STEPS = int(os.environ.get("FASHN_STEPS", "12"))   # env FASHN_STEPS overrides
MAX_STEPS = 50
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "15"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}

# Ephemeral (unsaved) results are kept this long so the results page works,
# then cleaned up on startup. Saved history items are kept indefinitely.
EPHEMERAL_RESULT_TTL_HOURS = int(os.environ.get("EPHEMERAL_RESULT_TTL_HOURS", "24"))

for d in (UPLOAD_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
