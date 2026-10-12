"""Manuscript layout region detection (header, footer, main_text, side_text, filler)."""
from .config import Config
from .pipeline import LayoutDetector
from .types import DetectionResult, Region, LABELS

__all__ = ["Config", "LayoutDetector", "DetectionResult", "Region", "LABELS"]
__version__ = "1.0.0"
