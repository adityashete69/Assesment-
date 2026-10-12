"""'filler' detectors: English overlay text (digitisation banners, stamps) and binding holes.

OCR is used *only* to recognise Latin-script text. Manuscript scripts (Devanagari, Grantha, Telugu ...)
come out as low-confidence garbage and are rejected by the confidence / alphabet filters.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

import cv2
import numpy as np

from .config import Config
from .geometry import Box, clip_box, union
from .preprocess import estimate_background
from .types import Region

log = logging.getLogger(__name__)
_WORD = re.compile(r"[A-Za-z]{2,}")


def tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def _ocr_views(gray: np.ndarray, cfg: Config) -> List[np.ndarray]:
    """Two renderings of a strip: raw gray, and background-flattened (dark text on white).

    Flattening rescues text on coloured / uneven backgrounds (e.g. the blue scanner mat)."""
    bg = estimate_background(gray, cfg.bg_median_kernel_frac)
    diff = np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
    flat = 255 - cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return [gray, flat]


def detect_overlay_text(bgr: np.ndarray, cfg: Config) -> List[Region]:
    """Find English text lines in the top/bottom bands of the image. Boxes are in `bgr` pixels."""
    if not cfg.use_ocr:
        return []
    if not tesseract_available():
        log.warning("Tesseract not found - English overlay text will not be detected. "
                    "Install it (apt-get install tesseract-ocr) or pass --no-ocr to silence this.")
        return []
    import pytesseract

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    band = max(20, int(cfg.ocr_band_frac * h))
    bands: List[Tuple[int, int]] = [(0, band), (max(band, h - band), h)] if 2 * band < h else [(0, h)]

    words: List[Tuple[Box, str, float]] = []
    weak: List[Tuple[Box, str, float]] = []
    for y0, y1 in bands:
        for view in _ocr_views(gray[y0:y1], cfg):
            try:
                d = pytesseract.image_to_data(view, lang=cfg.ocr_lang, config="--psm 11",
                                              output_type=pytesseract.Output.DICT)
            except Exception as exc:  # noqa: BLE001
                log.warning("OCR failed on a strip: %s", exc)
                continue
            for txt, conf, x, y, ww, hh in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
                conf = float(conf)
                if not txt.strip() or hh >= 0.25 * (y1 - y0) + 40:
                    continue
                item = ((x, y + y0, x + ww, y + y0 + hh), txt.strip(), conf)
                if conf >= cfg.ocr_min_conf and _WORD.search(txt):
                    words.append(item)
                elif conf >= 25:
                    weak.append(item)      # e.g. 'CC-0.' - only kept if it sits next to confirmed text

    return _group_words_into_lines(words, weak, w, h, cfg)


def _v_overlap(a: Box, b: Box) -> float:
    ov = min(a[3], b[3]) - max(a[1], b[1])
    return ov / max(1.0, min(a[3] - a[1], b[3] - b[1]))


def _h_gap(a: Box, b: Box) -> int:
    return max(a[0] - b[2], b[0] - a[2])


def _group_words_into_lines(words, weak, w: int, h: int, cfg: Config) -> List[Region]:
    """Merge words that share a text row (vertical overlap) and are close horizontally."""
    lines: List[dict] = []
    for box, txt, conf in sorted(words, key=lambda t: t[0][0]):      # left -> right
        bh = box[3] - box[1]
        for ln in lines:
            lh = ln["box"][3] - ln["box"][1]
            if _v_overlap(box, ln["box"]) >= 0.5 and _h_gap(box, ln["box"]) < 6 * max(bh, lh):
                ln["box"] = union(ln["box"], box)
                key = txt.lower()
                ln["words"][key] = max(conf, ln["words"].get(key, 0))
                break
        else:
            lines.append({"box": box, "words": {txt.lower(): conf}})

    out: List[Region] = []
    for ln in lines:
        confs = list(ln["words"].values())
        longest = max(len(t) for t in ln["words"])
        # a single stray token must be very confident and word-like; real banners have several words
        if not (len(confs) >= 2 or (longest >= 5 and max(confs) >= 90)):
            continue
        box = ln["box"]
        for wbox, _, _ in weak:                                      # attach neighbouring weak words
            if _v_overlap(wbox, box) >= 0.5 and _h_gap(wbox, box) < 2 * (box[3] - box[1]):
                box = union(box, wbox)
        p = cfg.ocr_pad
        x0, y0, x1, y1 = box
        out.append(Region("filler", clip_box((x0 - p, y0 - p, x1 + p, y1 + p), w, h),
                          float(np.mean(confs)) / 100.0, "ocr_overlay"))
    return out


def holes_to_regions(holes: List[Box], scale: float, w: int, h: int) -> List[Region]:
    out = []
    for (x0, y0, x1, y1) in holes:
        box = clip_box((x0 / scale, y0 / scale, x1 / scale, y1 / scale), w, h)
        bw, bh = x1 - x0, y1 - y0
        roundness = 1.0 - min(1.0, abs(bw - bh) / max(bw, bh, 1))
        out.append(Region("filler", box, 0.55 + 0.4 * roundness, "binding_hole"))
    return out
