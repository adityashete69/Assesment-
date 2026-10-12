"""Plain data containers shared by every module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixel coordinates

LABELS = ("header", "footer", "main_text", "side_text", "filler")


@dataclass
class Region:
    label: str
    bbox: Box
    confidence: float
    source: str = "rules"  # which detector produced it, e.g. "ocr_overlay", "block_geometry"

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
        }


@dataclass
class DetectionResult:
    width: int
    height: int
    page_bbox: Box
    regions: List[Region] = field(default_factory=list)
    skew_deg: float = 0.0

    def to_dict(self, image_name: str) -> Dict:
        return {
            "image": image_name,
            "width": self.width,
            "height": self.height,
            "page_bbox": [int(v) for v in self.page_bbox],
            "skew_deg": round(float(self.skew_deg), 2),
            "predictions": [r.to_dict() for r in self.regions],
        }
