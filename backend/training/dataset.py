"""VITON-HD dataset loader for FASHN VTON v1.5 LoRA fine-tuning.

Free, deterministic preprocessing (NO models needed at load time):
- LIP parse (`image-parse-v3`) -> upper-cloth(orange) + arms(light-blue) are
  erased to mid-gray (127) = FASHN's create_clothing_agnostic_image() equivalent.
  Background and identity regions (face/hair/lower body) are untouched.
- openpose_json -> DWPose-format bodies, drawn with FASHN's draw_pose().
  The garment pose is the flat-lay dummy (same as inference).

License: VITON-HD is CC BY-NC-SA 4.0 — research use only (docs/TRAINING.md).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "src"))

from fashn_vton.preprocessing.transforms import AspectPreserveResize, ResizePad  # noqa: E402
from fashn_vton.utils.tensor import numpy_to_torch  # noqa: E402
from fashn_vton.utils.keypoints import get_dummy_dw_keypoints  # noqa: E402
from fashn_vton.dwpose import draw_pose  # noqa: E402

CANVAS = 864
MODEL_W, MODEL_H = 576, 864
CATEGORY_TOP = 1

# LIP colour map (RGB), parse-v3 of VITON-HD
RED = (254, 0, 0)    # hair            -> keep (identity)
BLUE = (0, 0, 254)   # face            -> keep (identity)
TEAL = (0, 85, 85)   # lower body      -> keep (not covered by tops)
KEEP_COLORS = [RED, BLUE, TEAL]
ERASE_TOL = 40


def _find_split_root(root: Path, split: str) -> Path:
    for cand in Path(root).rglob("image"):
        if cand.parent.name == split:
            return cand.parent
    raise FileNotFoundError(f"No {split}/image found under {root}")


def _stems(split_dir: Path):
    stems = sorted(p.stem for p in (split_dir / "image").glob("*.jpg"))
    needed = {p.stem for p in (split_dir / "cloth").glob("*.jpg")}
    parse_ok = {p.stem for p in (split_dir / "image-parse-v3").glob("*.png")}
    pose_ok = {p.name[: -len("_keypoints.json")] for p in (split_dir / "openpose_json").glob("*_keypoints.json")}
    return [s for s in stems if s in needed and s in parse_ok and s in pose_ok]


def _to_tensor_m11(np_img: np.ndarray) -> torch.Tensor:
    return numpy_to_torch(np_img).float() / 127.5 - 1.0


def _to_tensor_gray(p: np.ndarray) -> torch.Tensor:
    t = numpy_to_torch(p).float()
    if t.ndim == 3:
        t = t[:1]
    else:
        t = t.unsqueeze(0)
    return t / 127.5 - 1.0


def _openpose_to_dwpose(json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    people = data.get("people") or []
    if not people:
        return get_dummy_dw_keypoints()
    arr = np.array(people[0].get("pose_keypoints_2d", []), dtype=np.float32).reshape(-1, 3)
    candidate = arr[:18, :2].copy()
    subset = np.arange(18, dtype=np.float32)[None, :]
    return {"bodies": {"candidate": candidate, "subset": subset}}


def _erase_garment(person_np: np.ndarray, parse_np: np.ndarray) -> np.ndarray:
    """Fallback agnostic image when a FASHN-parser cache is unavailable.
    Close approximation (keeps hair/face/lower-body, erases the rest of the
    labeled person) — use scripts/cache_ca_images.py for exact FASHN parity."""
    ca = person_np.copy()
    p = parse_np.astype(np.int16)
    labeled = np.any(p != 0, axis=-1)          # person pixels (background is pure black)
    keep = np.zeros(parse_np.shape[:2], dtype=bool)
    for color in KEEP_COLORS:
        dist = np.abs(p - np.array(color, dtype=np.int16)).sum(axis=-1)
        keep |= dist < ERASE_TOL * 3
    ca[labeled & ~keep] = 127
    return ca


class VitonHDDataset(Dataset):
    """Upper-body garments (category=tops). GT is the person wearing the garment."""

    def __init__(self, root, split: str = "train", limit: int | None = None):
        self.root = Path(root)
        self.split_dir = _find_split_root(self.root, split)
        self.stems = _stems(self.split_dir)
        if limit:
            self.stems = self.stems[:limit]
        self.pre_resize_pil = AspectPreserveResize(target_size=(CANVAS, CANVAS), mode="fit", backend="pil")
        self.resize_pad = ResizePad((MODEL_W, MODEL_H), backend="opencv")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        person = Image.open(self.split_dir / "image" / f"{stem}.jpg").convert("RGB")
        cloth = Image.open(self.split_dir / "cloth" / f"{stem}.jpg").convert("RGB")

        # ---- identical pre_resize as the FASHN pipeline (PIL, LANCZOS) ----
        w0, _ = person.size
        person = self.pre_resize_pil(person, allow_upsampling=False)
        scale = person.size[0] / w0
        cloth = self.pre_resize_pil(cloth, allow_upsampling=False)

        person_np = np.array(person)
        cloth_np = np.array(cloth)

        pose = _openpose_to_dwpose(self.split_dir / "openpose_json" / f"{stem}_keypoints.json")
        pose["bodies"]["candidate"] = pose["bodies"]["candidate"] * scale
        person_pose_img = draw_pose(pose, person_np.shape[0], person_np.shape[1], grayscale=True)

        ca_path = self.root / "processed" / "train" / f"{stem}_ca.png"
        if ca_path.exists():
            # pre-computed with FASHN's own parser (guaranteed same distribution)
            ca_image = np.array(Image.open(ca_path).convert("RGB"))
            assert ca_image.shape[:2] == person_np.shape[:2], f"cached ca size {ca_image.shape} != person {person_np.shape}"
        else:
            parse = Image.open(self.split_dir / "image-parse-v3" / f"{stem}.png").convert("RGB")
            parse = parse.resize((person.size[0], person.size[1]), Image.NEAREST)
            ca_image = _erase_garment(person_np, np.array(parse))
        garment_image = cloth_np.copy()
        garment_pose_img = draw_pose(get_dummy_dw_keypoints(), cloth_np.shape[0], cloth_np.shape[1], grayscale=True)

        # ---- ResizePad into model canvas (same as pipeline) ----
        ca_image = self.resize_pad(ca_image, mem_padding=True)
        garment_image = self.resize_pad(garment_image)
        gt_image = self.resize_pad(person_np, mem_padding=True)
        person_pose_img = self.resize_pad(person_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)
        garment_pose_img = self.resize_pad(garment_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)

        return {
            "name": stem,
            "gt": _to_tensor_m11(gt_image),
            "ca_image": _to_tensor_m11(ca_image),
            "garment_image": _to_tensor_m11(garment_image),
            "person_pose": _to_tensor_gray(person_pose_img),
            "garment_pose": _to_tensor_gray(garment_pose_img),
            "category": torch.tensor(CATEGORY_TOP, dtype=torch.long),
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch] if k == "name" else torch.stack([b[k] for b in batch])
    return out
