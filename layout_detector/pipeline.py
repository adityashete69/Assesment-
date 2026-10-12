"""End-to-end detector: image (BGR ndarray) -> DetectionResult."""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from .classify import classify_blocks
from .config import Config
from .filler import detect_overlay_text, holes_to_regions
from .geometry import Box, clip_box, scale_box
from .preprocess import (estimate_skew, extract_ink, median_glyph_height,
                         resize_for_work, rotation_matrix)
from .segmentation import find_blocks, merge_lines_into_paragraphs
from .types import DetectionResult, Region

log = logging.getLogger(__name__)

_ORDER = {"header": 0, "main_text": 1, "side_text": 2, "footer": 3, "filler": 4}


def _map_box_back(box: Box, inv: np.ndarray) -> Box:
    """Axis-aligned bounding box of a rotated-back rectangle."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)
    out = cv2.transform(pts, inv).reshape(-1, 2)
    return (int(out[:, 0].min()), int(out[:, 1].min()), int(out[:, 0].max()), int(out[:, 1].max()))


class LayoutDetector:
    """Classical (training-free) manuscript layout detector.

    Stages: page localisation -> illumination-robust ink extraction -> filler detection
    (OCR overlay text, binding holes) -> deskew -> block segmentation -> geometric classification.
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        cfg = self.cfg
        H, W = bgr.shape[:2]
        work, scale = resize_for_work(bgr, cfg.work_max_side)
        wh, ww = work.shape[:2]

        # 1) page + ink
        res = extract_ink(work, cfg)
        ink, diff = res.ink.copy(), res.diff

        # 2) filler: English overlay text (OCR on full-resolution strips) + binding holes
        regions: List[Region] = detect_overlay_text(bgr, cfg)
        for r in regions:
            x0, y0, x1, y1 = [int(round(v * scale)) for v in r.bbox]
            pad = 3
            ink[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad] = 0
        regions += holes_to_regions(res.holes, scale, W, H)

        # 3) deskew (analysis only; boxes are mapped back to the original frame)
        angle = 0.0
        inv = None
        page_work = res.page_bbox
        if cfg.deskew:
            a = estimate_skew(ink, cfg.max_skew_deg)
            if abs(a) >= cfg.min_skew_deg:
                m = rotation_matrix(ink.shape, a)
                ink = cv2.warpAffine(ink, m, (ww, wh), flags=cv2.INTER_NEAREST)
                diff = cv2.warpAffine(diff, m, (ww, wh), flags=cv2.INTER_LINEAR)
                inv = cv2.invertAffineTransform(m)
                px0, py0, px1, py1 = res.page_bbox
                corners = np.array([[px0, py0], [px1, py0], [px1, py1], [px0, py1]], np.float32).reshape(-1, 1, 2)
                rc = cv2.transform(corners, m).reshape(-1, 2)
                page_work = (int(max(0, rc[:, 0].min())), int(max(0, rc[:, 1].min())),
                             int(min(ww, rc[:, 0].max())), int(min(wh, rc[:, 1].max())))
                angle = a

        # 4) blocks + classification
        glyph_h = median_glyph_height(ink, cfg.min_component_area, 0.1 * wh)
        blocks = merge_lines_into_paragraphs(find_blocks(ink, diff, glyph_h, cfg), glyph_h)
        text_regions = classify_blocks(blocks, page_work, glyph_h, cfg)

        # 5) back to original coordinates, clip to the page
        for r in text_regions:
            box = _map_box_back(r.bbox, inv) if inv is not None else r.bbox
            r.bbox = clip_box(scale_box(box, 1.0 / scale), W, H)
        regions += text_regions

        regions.sort(key=lambda r: (_ORDER[r.label], r.bbox[1], r.bbox[0]))
        page = clip_box(scale_box(res.page_bbox, 1.0 / scale), W, H)
        if res.page_bbox[2] - res.page_bbox[0] >= ww - 1 and res.page_bbox[3] - res.page_bbox[1] >= wh - 1:
            page = (0, 0, W - 1, H - 1)
        return DetectionResult(W, H, page, regions, angle)
