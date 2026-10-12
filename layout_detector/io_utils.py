"""Reading images, discovering inputs and writing outputs. Inputs are only ever *read*."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_image_bgr(path: Path) -> np.ndarray:
    """Load any common image as 3-channel BGR uint8 (alpha is composited onto white)."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        rgb = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def list_images(input_path: Path, recursive: bool = True) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTS else []
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
