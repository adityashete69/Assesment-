"""Integration tests on the two sample pages from the assignment (need Tesseract for the OCR part)."""
from pathlib import Path

import pytest

from layout_detector import Config, LayoutDetector
from layout_detector.filler import tesseract_available
from layout_detector.io_utils import load_image_bgr

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
pytestmark = pytest.mark.skipif(not SAMPLES.exists(), reason="sample images missing")


def _labels(res):
    return [r.label for r in res.regions]


def test_samples_without_ocr_find_main_text():
    det = LayoutDetector(Config(use_ocr=False))
    for f in sorted(SAMPLES.glob("*.png")):
        res = det.detect(load_image_bgr(f))
        assert "main_text" in _labels(res), f.name


def test_palm_leaf_binding_hole_is_filler():
    res = LayoutDetector(Config(use_ocr=False)).detect(load_image_bgr(SAMPLES / "sample_palm_leaf.png"))
    assert any(r.source == "binding_hole" for r in res.regions)


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_samples_overlay_banners_are_filler():
    det = LayoutDetector(Config())
    for f in sorted(SAMPLES.glob("*.png")):
        res = det.detect(load_image_bgr(f))
        banners = [r for r in res.regions if r.source == "ocr_overlay"]
        assert len(banners) == 2, f.name                     # one at the top, one at the bottom
        h = res.height
        assert min(r.bbox[1] for r in banners) < 0.15 * h
        assert max(r.bbox[3] for r in banners) > 0.85 * h
        # the banner must not be swallowed by main_text
        main = [r for r in res.regions if r.label == "main_text"][0]
        assert all(b.bbox[1] >= main.bbox[3] - 5 or b.bbox[3] <= main.bbox[1] + 5 for b in banners)
