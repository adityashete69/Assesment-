"""Group cleaned ink into text *blocks* (lines/paragraphs/columns) using adaptive dilation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .config import Config
from .geometry import Box


@dataclass
class Block:
    bbox: Box            # tight box around the block's ink pixels
    ink_px: int          # number of ink pixels
    contrast: float      # mean ink darkness relative to local background (0-255)

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def w(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> int:
        return self.bbox[3] - self.bbox[1]


def find_blocks(ink: np.ndarray, diff: np.ndarray, glyph_h: float, cfg: Config) -> List[Block]:
    """Dilate ink by amounts proportional to glyph height so characters -> words -> lines -> blocks.

    Using glyph height (not fixed pixels) makes the merge distance script- and scale-independent."""
    kx = max(3, int(cfg.block_kx * glyph_h)) | 1
    ky = max(3, int(cfg.block_ky * glyph_h)) | 1
    merged = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, labels, _, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    ys, xs = np.nonzero(ink)
    lab = labels[ys, xs]
    blocks: List[Block] = []
    for i in range(1, n):
        sel = lab == i
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        bx, by = xs[sel], ys[sel]
        if cnt >= 400:   # robust extent: ignore the outermost 0.5% of ink (edge specks, stray dots)
            x0, x1 = np.percentile(bx, [0.5, 99.5])
            y0, y1 = np.percentile(by, [0.5, 99.5])
        else:
            x0, x1, y0, y1 = bx.min(), bx.max(), by.min(), by.max()
        pad = int(round(0.25 * glyph_h))   # undo the quantile trim at glyph level
        box = (int(x0) - pad, int(y0) - pad, int(x1) + 1 + pad, int(y1) + 1 + pad)
        contrast = float(diff[by, bx].mean())
        blocks.append(Block(box, cnt, contrast))

    if not blocks:
        return []
    biggest = max(b.ink_px for b in blocks)
    min_px = max(cfg.min_block_ink_px, 0.4 * glyph_h * glyph_h)   # smaller than ~one glyph = speck
    return [b for b in blocks if b.ink_px >= min_px and b.ink_px >= cfg.min_block_ink_frac * biggest]


def merge_lines_into_paragraphs(blocks: List[Block], glyph_h: float) -> List[Block]:
    """Second-level grouping: stack text lines into paragraphs / columns.

    Two blocks are merged when they overlap horizontally (>=50% of the narrower one) and the vertical
    gap between them is <= 1.5 x the page's *own* median line gap (>= one glyph height). A header or
    footer is only separated if its gap is clearly larger than the normal line spacing, so widely
    spaced palm-leaf lines are not mistaken for header/footer."""
    if len(blocks) < 2:
        return list(blocks)

    def h_overlap(a: Block, b: Block) -> float:
        ov = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
        return ov / max(1.0, min(a.w, b.w))

    def v_gap(a: Block, b: Block) -> int:
        return max(a.bbox[1] - b.bbox[3], b.bbox[1] - a.bbox[3])

    order = sorted(range(len(blocks)), key=lambda i: blocks[i].bbox[1])
    gaps = []
    for ia, ib in zip(order, order[1:]):
        a, b = blocks[ia], blocks[ib]
        if h_overlap(a, b) >= 0.5 and v_gap(a, b) >= 0:
            gaps.append(v_gap(a, b))
    pitch = float(np.median(gaps)) if len(gaps) >= 3 else glyph_h
    thresh = max(1.5 * pitch, glyph_h)

    parent = list(range(len(blocks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if h_overlap(blocks[i], blocks[j]) >= 0.5 and v_gap(blocks[i], blocks[j]) <= thresh:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(len(blocks)):
        groups.setdefault(find(i), []).append(blocks[i])
    merged = []
    for members in groups.values():
        ink = sum(m.ink_px for m in members)
        box = (min(m.bbox[0] for m in members), min(m.bbox[1] for m in members),
               max(m.bbox[2] for m in members), max(m.bbox[3] for m in members))
        contrast = sum(m.contrast * m.ink_px for m in members) / max(1, ink)
        merged.append(Block(box, ink, contrast))
    return merged
