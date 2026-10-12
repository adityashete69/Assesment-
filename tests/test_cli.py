import hashlib
import json
from pathlib import Path

import cv2
import pytest

import inference
from tests.synth import make_page


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def dataset(tmp_path):
    inp = tmp_path / "in"
    (inp / "sub").mkdir(parents=True)
    for i, d in enumerate([inp, inp, inp / "sub"]):
        img, _ = make_page(seed=i)
        cv2.imwrite(str(d / f"page{i}.png"), img)
    (inp / "notes.txt").write_text("ignore me")
    return inp


def test_cli_batch_outputs_and_inputs_unchanged(dataset, tmp_path):
    out = tmp_path / "out"
    before = {p: sha(p) for p in dataset.rglob("*") if p.is_file()}
    rc = inference.run(["--input", str(dataset), "--output", str(out), "--no-ocr"])
    assert rc == 0
    assert {p: sha(p) for p in dataset.rglob("*") if p.is_file()} == before     # inputs untouched

    summary = json.loads((out / "predictions.json").read_text())
    assert summary["num_images"] == 3 and summary["num_failed"] == 0
    assert len(list((out / "annotated").glob("*.png"))) == 3
    assert len(list((out / "json").glob("*.json"))) == 3
    assert (out / "json" / "sub__page2.json").exists()          # sub-folder names don't collide

    rec = summary["results"][0]
    assert {"image", "width", "height", "page_bbox", "skew_deg", "predictions"} <= set(rec)
    for p in rec["predictions"]:
        assert set(p) == {"label", "bbox", "confidence", "source"}
        x0, y0, x1, y1 = p["bbox"]
        assert 0 <= x0 < x1 < rec["width"] and 0 <= y0 < y1 < rec["height"]


def test_cli_annotated_image_has_same_size(dataset, tmp_path):
    out = tmp_path / "out"
    inference.run(["--input", str(dataset / "page0.png"), "--output", str(out), "--no-ocr"])
    ann = cv2.imread(str(out / "annotated" / "page0_annotated.png"))
    orig = cv2.imread(str(dataset / "page0.png"))
    assert ann.shape == orig.shape


def test_cli_survives_corrupt_file(dataset, tmp_path):
    (dataset / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "out"
    rc = inference.run(["--input", str(dataset), "--output", str(out), "--no-ocr"])
    assert rc == 1
    summary = json.loads((out / "predictions.json").read_text())
    assert summary["num_failed"] == 1 and len(summary["results"]) == 3


def test_cli_errors(tmp_path):
    assert inference.run(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inference.run(["--input", str(empty), "--output", str(tmp_path / "o")]) == 2


def test_cli_output_inside_input_is_not_reprocessed(dataset):
    out = dataset / "results"
    inference.run(["--input", str(dataset), "--output", str(out), "--no-ocr"])
    rc = inference.run(["--input", str(dataset), "--output", str(out), "--no-ocr"])
    assert rc == 0
    assert json.loads((out / "predictions.json").read_text())["num_images"] == 3


def test_load_image_handles_alpha(tmp_path):
    import numpy as np
    from layout_detector.io_utils import load_image_bgr
    rgba = np.zeros((10, 12, 4), np.uint8)       # fully transparent -> white
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), rgba)
    img = load_image_bgr(p)
    assert img.shape == (10, 12, 3) and img.min() == 255
