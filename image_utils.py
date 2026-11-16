#Foundational image loading, for images
"""Shared image utilities."""
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(path_or_bytes, max_side: int = 1400) -> np.ndarray:
    """Load an image as RGB numpy array, honoring EXIF orientation."""
    if isinstance(path_or_bytes, (str, Path)):
        img = Image.open(path_or_bytes)
    else:
        img = Image.open(io.BytesIO(path_or_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.array(img)


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def np_to_b64(arr: np.ndarray, fmt: str = ".png") -> str:
    ok, buf = cv2.imencode(fmt, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())
