"""Rule-based assignment of layout classes to text blocks.

The *largest-ink block* is the anchor ("body core"). Every other block is judged by where it sits
relative to that core and the page:

    above core, vertically separate, in the top band      -> header
    below core, vertically separate, in the bottom band   -> footer
    beside core (vertical overlap), outer page margin     -> side_text   (or main_text if heavy = 2nd column)
    much fainter than body ink, outside the body          -> filler      (pencil / faint annotation)
    everything else                                       -> main_text
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import Config
from .geometry import Box, overlap_1d, union
from .segmentation import Block
from .types import Region


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def classify_blocks(blocks: List[Block], page: Box, glyph_h: float, cfg: Config) -> List[Region]:
    if not blocks:
        return []
    px0, py0, px1, py1 = page
    pw, ph = max(1, px1 - px0), max(1, py1 - py0)
    total_ink = float(sum(b.ink_px for b in blocks))
    core = max(blocks, key=lambda b: b.ink_px)
    cx0, cy0, cx1, cy1 = core.bbox
    body_contrast = max(core.contrast, 1.0)

    regions: List[Region] = []
    for b in blocks:
        x0, y0, x1, y1 = b.bbox
        if b is core:
            conf = 0.60 + 0.35 * _clip01(b.ink_px / total_ink)
            regions.append(Region("main_text", b.bbox, conf, "block_geometry"))
            continue

        v_overlap = overlap_1d(y0, y1, cy0, cy1) / max(1, min(b.h, cy1 - cy0))
        h_overlap = overlap_1d(x0, x1, cx0, cx1) / max(1, min(b.w, cx1 - cx0))
        heavy = b.ink_px >= cfg.second_column_ink_frac * core.ink_px
        faint = b.contrast < cfg.faint_contrast_ratio * body_contrast

        above = y1 <= cy0 + 0.25 * b.h and v_overlap < 0.2
        below = y0 >= cy1 - 0.25 * b.h and v_overlap < 0.2
        top_band = (b.cy - py0) <= cfg.header_band_frac * ph
        bot_band = (py1 - b.cy) <= cfg.footer_band_frac * ph
        side_band = (b.cx - px0) <= cfg.side_band_frac * pw or (px1 - b.cx) <= cfg.side_band_frac * pw
        beside = v_overlap >= 0.2 and h_overlap < 0.5

        if faint and not heavy:
            # pencil strokes / faint annotations are explicitly 'filler' in the task definition
            conf = 0.45 + 0.4 * _clip01(1.0 - b.contrast / (cfg.faint_contrast_ratio * body_contrast))
            regions.append(Region("filler", b.bbox, conf, "faint_text"))
        elif above and top_band and not heavy:
            gap = (cy0 - y1) / max(1.0, 1.5 * glyph_h)
            depth = 1.0 - (b.cy - py0) / (cfg.header_band_frac * ph)
            conf = 0.45 + 0.30 * _clip01(gap) + 0.20 * _clip01(depth)
            regions.append(Region("header", b.bbox, conf, "block_geometry"))
        elif below and bot_band and not heavy:
            gap = (y0 - cy1) / max(1.0, 1.5 * glyph_h)
            depth = 1.0 - (py1 - b.cy) / (cfg.footer_band_frac * ph)
            conf = 0.45 + 0.30 * _clip01(gap) + 0.20 * _clip01(depth)
            regions.append(Region("footer", b.bbox, conf, "block_geometry"))
        elif beside and heavy:
            regions.append(Region("main_text", b.bbox, 0.55 + 0.3 * _clip01(b.ink_px / total_ink), "block_geometry"))
        elif beside and side_band:
            sep = (cx0 - x1 if x1 <= cx0 else x0 - cx1) / max(1.0, 2.0 * glyph_h)
            conf = 0.45 + 0.30 * _clip01(sep) + 0.20 * (1.0 - _clip01(b.ink_px / core.ink_px))
            regions.append(Region("side_text", b.bbox, conf, "block_geometry"))
        else:
            # detached but not clearly header/footer/margin: most likely another body paragraph
            regions.append(Region("main_text", b.bbox, 0.40 + 0.3 * _clip01(b.ink_px / total_ink), "block_geometry"))
    return regions
