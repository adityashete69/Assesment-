import numpy as np
import pytest

from layout_detector import Config, LayoutDetector, LABELS
from layout_detector.geometry import iou
from tests.synth import degrade, make_page, rotate


@pytest.fixture(scope="module")
def detector():
    return LayoutDetector(Config(use_ocr=False))


def by_label(result):
    out = {}
    for r in result.regions:
        out.setdefault(r.label, []).append(r)
    return out


def test_all_classes_found_on_synthetic_page(detector):
    img, exp = make_page()
    labels = by_label(detector.detect(img))
    for lab in ("header", "footer", "main_text", "side_text"):
        assert lab in labels, f"missing {lab}"
    best = max(iou(r.bbox, exp["main_text"]) for r in labels["main_text"])
    assert best > 0.8
    assert iou(labels["header"][0].bbox, exp["header"]) > 0.5
    assert iou(labels["footer"][0].bbox, exp["footer"]) > 0.5


def test_no_header_footer_when_absent(detector):
    img, _ = make_page(header=False, footer=False, side=False)
    labels = by_label(detector.detect(img))
    assert set(labels) == {"main_text"}


def test_side_text_is_left_of_body(detector):
    img, exp = make_page()
    res = detector.detect(img)
    body_x0 = max(r.bbox[0] for r in res.regions if r.label == "main_text")
    sides = [r for r in res.regions if r.label == "side_text"]
    assert sides and all(r.bbox[2] <= body_x0 + 5 for r in sides)


def test_boxes_inside_page_and_confidence_valid(detector):
    img, _ = make_page()
    res = detector.detect(img)
    h, w = img.shape[:2]
    for r in res.regions:
        x0, y0, x1, y1 = r.bbox
        assert 0 <= x0 < x1 <= w - 1 and 0 <= y0 < y1 <= h - 1
        assert 0.0 <= r.confidence <= 1.0
        assert r.label in LABELS


@pytest.mark.parametrize("angle", [-3.0, 2.5, 4.0])
def test_skew_robust(detector, angle):
    img, exp = make_page()
    res = detector.detect(rotate(img, angle))
    labels = by_label(res)
    assert {"main_text", "header", "footer"} <= set(labels)
    assert abs(res.skew_deg + angle) <= 1.0          # estimate undoes the applied rotation
    h, w = img.shape[:2]
    for r in res.regions:
        assert 0 <= r.bbox[0] < r.bbox[2] <= w - 1 and 0 <= r.bbox[1] < r.bbox[3] <= h - 1


@pytest.mark.parametrize("kw", [
    dict(blur=1.5),                          # out of focus
    dict(contrast=0.55),                     # faded ink
    dict(gradient=0.35),                     # uneven lighting
    dict(noise=12.0),                        # sensor noise / grain
    dict(blur=1.0, contrast=0.6, gradient=0.25, noise=8.0),   # all together
])
def test_degradation_robust(detector, kw):
    img, exp = make_page()
    res = detector.detect(degrade(img, **kw))
    mains = [r for r in res.regions if r.label == "main_text"]
    assert mains and max(iou(r.bbox, exp["main_text"]) for r in mains) > 0.6


def test_palm_leaf_like_page_on_coloured_background(detector):
    """Page = pale leaf on a blue mat: the detector must localise the leaf and ignore the mat."""
    import cv2
    img, _ = make_page(width=1400, height=700, header=False, footer=False, side=False, n_lines=6)
    mat = np.full((900, 1600, 3), (150, 110, 80), np.uint8)       # BGR blue-ish
    mat[100:800, 100:1500] = img[:700, :1400]
    res = LayoutDetector(Config(use_ocr=False)).detect(mat)
    x0, y0, x1, y1 = res.page_bbox
    assert abs(x0 - 100) < 40 and abs(y0 - 100) < 40 and abs(x1 - 1500) < 40 and abs(y1 - 800) < 40
    assert all(r.bbox[0] >= 100 - 5 for r in res.regions if r.label != "filler")


def test_blank_page_returns_no_regions(detector):
    res = detector.detect(np.full((600, 800, 3), 220, np.uint8))
    assert res.regions == []


def test_tiny_and_grayscale_like_images_do_not_crash(detector):
    detector.detect(np.full((20, 30, 3), 128, np.uint8))
    img, _ = make_page()
    g = np.repeat(np.mean(img, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
    assert any(r.label == "main_text" for r in detector.detect(g).regions)


def test_input_array_not_modified(detector):
    img, _ = make_page()
    before = img.copy()
    detector.detect(img)
    assert np.array_equal(img, before)


def test_missing_tesseract_degrades_gracefully(monkeypatch, caplog):
    import layout_detector.filler as filler
    monkeypatch.setattr(filler, "tesseract_available", lambda: False)
    img, _ = make_page()
    res = LayoutDetector(Config(use_ocr=True)).detect(img)
    assert any(r.label == "main_text" for r in res.regions)
    assert "Tesseract not found" in caplog.text
