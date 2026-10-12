# Manuscript Layout Region Detection

End-to-end Python pipeline that finds and classifies layout regions on historical manuscript pages
(palm-leaf and paper): **header, footer, main_text, side_text, filler**.
It writes annotated images plus JSON predictions (bounding box, label, confidence) for single images or whole folders.

```
python inference.py --input ./data/samples --output ./results
```

![palm leaf](results_example/annotated/sample_palm_leaf_annotated.png)
![paper](results_example/annotated/sample_paper_annotated.png)


### All five classes in one picture
The two assignment samples contain only body text plus banners/hole, so they correctly yield only `main_text` and `filler`.
`data/demo/demo_all_five_classes.png` (real handwriting crops placed as header, footer and margin notes, plus English banners)
shows every label firing: output in `results_example_demo/`.

![demo](results_example_demo/annotated/demo_all_five_classes_annotated.png)

---------------------------------------------------------------------------------------------------

## 1. Setup

Python >= 3.9. One system package (Tesseract) is used only to recognise the *English* overlay text that counts as `filler`.

```bash
# Linux / Colab
sudo apt-get install -y tesseract-ocr          # (Colab: !apt-get -qq install tesseract-ocr)
# macOS:  brew install tesseract

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No Tesseract? Everything still runs - English banners just won't be detected (a warning is logged; use `--no-ocr` to silence it).

## 2. Usage

```bash
python inference.py --input ./data/samples --output ./results        # folder (recursive)
python inference.py --input page.jpg --output ./results              # single file
python inference.py --input ./scans --output ./results --no-ocr --no-deskew --work-size 2000
```

| flag | meaning |
|---|---|
| `--input` | image file or folder (jpg, png, tif, bmp, webp) |
| `--output` | output folder (created) - may even live inside `--input`; its contents are never re-processed |
| `--no-recursive` | do not descend into sub-folders |
| `--no-ocr` | skip English-text detection |
| `--no-deskew` | disable skew estimation |
| `--work-size N` | analysis resolution, long side in px (default 1600) |

Exit codes: `0` all ok, `1` some files failed (the rest are still processed, errors listed in `predictions.json`), `2` bad input.

### Output layout
```
results/
  annotated/<name>_annotated.png   # copy of the page with coloured boxes + "label confidence"
  json/<name>.json                 # one file per page
  predictions.json                 # all pages + any errors
```
Sub-folder images are named `sub__page.json` so names never collide. **Inputs are opened read-only and never modified.**

```json
{
  "image": "sample_palm_leaf.png", "width": 2048, "height": 834,
  "page_bbox": [142, 78, 1779, 699], "skew_deg": 1.5,
  "predictions": [
    {"label": "main_text", "bbox": [316, 83, 1704, 671], "confidence": 0.95, "source": "block_geometry"},
    {"label": "filler",    "bbox": [700, 369, 778, 429], "confidence": 0.8582, "source": "binding_hole"},
    {"label": "filler",    "bbox": [575, 24, 1472, 59],  "confidence": 0.9375, "source": "ocr_overlay"}
  ]
}
```
`bbox = [x0, y0, x1, y1]` in pixels of the **original** image, always clipped inside the image.
`skew_deg` is the rotation applied *for analysis only* (boxes are mapped back). `source` says which detector fired.

## 3. Method and model rationale

**Why no deep model?** The assignment ships no labelled data (only two sample images), so a trained detector
(YOLO / DETR / LayoutLM-style) would have nothing honest to learn from and would be impossible to validate.
A deterministic, explainable pipeline is reproducible, runs on CPU in ~2 s/page, works for any script
(it never reads the characters), and can later *bootstrap* a learned model (see section 6).

```
image -> work copy (<=1600px)
  1. page localisation      border-colour model in Lab; "page == frame" fallbacks (flat scans, lighting gradients)
  2. ink extraction         dilate+median background estimate -> subtract (flattens lighting/stains/bleed-through)
                            -> Otsu with floor -> remove specks, ruling lines, big solid blobs
  3. filler detectors       a) OCR (Tesseract, eng) on top/bottom 15% strips, 2 image views, line grouping
                            b) binding holes: round, solid, mid-sized blobs
  4. deskew (analysis)      projection-profile angle search +-5 deg; boxes mapped back afterwards
  5. block segmentation     dilation scaled by median glyph height (script/scale independent) -> lines
  6. paragraph merge        stack lines using the page's own median line gap
  7. classification         geometry relative to the largest-ink block ("body core")
  8. post-process           map to original pixels, clip to image, sort, write JSON + annotated PNG
```

### How each class is decided
| class | rule |
|---|---|
| `main_text` | the largest-ink block, plus other heavy blocks (>=35% of its ink) beside it = extra columns |
| `header` | block fully **above** the core, separated by a clear gap, centred in the top 22% of the page |
| `footer` | same, below the core, bottom 22% |
| `side_text` | light block **beside** the core (vertical overlap, little horizontal overlap) in the outer 28% of the width |
| `filler` | English overlay text (OCR); binding holes; blocks much fainter than body ink (pencil / faint notes) |

### Confidence
Confidence is a **heuristic score in [0, 1], not a calibrated probability**. OCR filler uses mean word confidence;
holes use roundness; text regions combine ink share, gap to the body and depth inside the header/footer band.
Use it for ranking/filtering, not as a probability.

### Robustness
| condition | handling |
|---|---|
| faded ink, bleed-through, stains | background subtraction; adaptive floor; solid blobs removed |
| uneven lighting | median-of-dilated background; page-==-frame fallback for gradients |
| blur, noise | tested at sigma 1.5 blur and noise sigma 12 |
| skew | +-5 deg estimated and undone for analysis, boxes mapped back |
| palm leaf vs paper | leaf found from border colour; paper treated as full frame; ruling lines removed |
| any script / aspect ratio | geometry only; merge distances scale with measured glyph height |
| huge scans | analysis at <=1600px, results rescaled to original coordinates |

## 4. Tests
```bash
python -m pytest            # 30 tests, ~15 s
```
Synthetic seeded pages (`tests/synth.py`) cover every class, skew (-3, 2.5, 4 deg), blur / fade / lighting / noise,
leaf-on-coloured-mat page detection, blank & tiny images, box-in-bounds, JSON schema, CLI batch behaviour,
unchanged inputs (SHA-256), corrupt files, output-inside-input, RGBA loading, and missing Tesseract.
`tests/test_samples.py` checks the two assignment samples (both banners + the binding hole are `filler`).

## 5. Known limitations (please read)
* **Not evaluated on a labelled benchmark** - none was provided. Behaviour on the two samples is verified visually and by tests;
  the two samples contain no true header/footer/marginalia, so those rules are validated on composite pages built from
  real handwriting crops and synthetic pages only. Expect to tune thresholds in `config.py` on real data.
* English text is searched only in the top/bottom 15% of the image (where digitisation banners sit). A stamp mid-page is missed.
* Pencil text is only caught when it passes the ink threshold *and* is much fainter than the body ink.
* Multi-column pages work when columns are separate blocks; a column split inside a shared line is returned as one `main_text`.
* Curved / warped lines (page curl) are not rectified; heavy warping widens boxes.
* Header/footer vs. body relies on a visible gap. Running titles tightly attached to the body will merge into `main_text`.

## 6. Extending
`LayoutDetector.detect(bgr) -> DetectionResult` is the only interface the CLI uses. To plug in a trained model,
implement the same method (e.g. YOLOv8 fine-tuned on corrected output of this pipeline: run it as a pseudo-labeller,
fix the boxes by hand, train, then compare both detectors with `iou()` in `geometry.py`).

## 7. Project layout
```
inference.py                CLI
layout_detector/
  config.py                 every threshold, documented
  preprocess.py             page mask, background/ink, ruling lines, skew
  filler.py                 OCR banners, binding holes
  segmentation.py           lines -> blocks -> paragraphs
  classify.py               geometric rules + confidence
  pipeline.py               LayoutDetector
  geometry.py io_utils.py visualize.py types.py
tests/                      pytest suite + synthetic page generator
data/samples/               the two sample pages from the assignment
results_example/            outputs for those samples
```
