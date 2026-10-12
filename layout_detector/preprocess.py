"""Image-level preprocessing: page localisation, illumination normalisation, ink extraction, skew.

Everything here works on a *working copy* (long side <= Config.work_max_side).
The original file is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import Config
from .geometry import Box


# --------------------------------------------------------------------------- helpers
def resize_for_work(bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Return (resized copy, scale) where scale = work_size / original_size (<= 1)."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return bgr, scale


def estimate_background(gray: np.ndarray, kernel_frac: float) -> np.ndarray:
    """Dilate + large-window median of a down-sampled image = smooth paper/leaf brightness.

    Handwriting strokes and small holes are thinner than the window, so they vanish from the
    estimate; subtracting it removes uneven lighting and gradients (illumination flattening).
    """
    h, w = gray.shape
    small_w = 400
    s = small_w / float(max(h, w)) if max(h, w) > small_w else 1.0
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    # Grey dilation first: dark pen strokes (a few low-res pixels wide) disappear, so even very
    # dense writing cannot drag the paper estimate down. Holes/stains much wider survive.
    small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    k = int(max(h, w) * s * kernel_frac) | 1
    k = max(k, 5)
    bg = cv2.medianBlur(small, k)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, k / 4.0))
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- page
def detect_page_mask(bgr: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Box]:
    """Separate the manuscript leaf/sheet from the scanning background.

    Border colour is taken as the background reference; pixels far from it in Lab space are
    'page'. If the page already fills the frame (flat-bed scans), the whole image is the page.
    """
    h, w = bgr.shape[:2]
    full = (0, 0, w, h)
    full_mask = np.full((h, w), 255, np.uint8)

    # Heavy blur: thin pen strokes melt into the paper colour, large colour regions survive.
    sigma = max(2.0, 0.008 * min(h, w))
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), sigma), cv2.COLOR_BGR2LAB).astype(np.float32)
    r = max(2, int(0.03 * min(h, w)))
    ring = np.concatenate([lab[:r].reshape(-1, 3), lab[-r:].reshape(-1, 3),
                           lab[:, :r].reshape(-1, 3), lab[:, -r:].reshape(-1, 3)])
    ref = np.median(ring, axis=0)
    dist = np.linalg.norm(lab - ref, axis=2)
    dist8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    thr, fg = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Page == frame when the centre of the image has the same colour as the border ring
    # (flat-bed scans, pages cropped to the paper). Otherwise the ring is scanner background.
    # Use a low percentile of the centre's colour distance so dense writing cannot fool it.
    centre_dist = dist[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if np.percentile(centre_dist, 25) < 10.0 or thr < 40:
        return full_mask, full

    k = max(3, int(0.02 * min(h, w))) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return full_mask, full
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    if bw * bh < cfg.page_min_area_frac * w * h:
        return full_mask, full
    # A real page is a solid blob; a sparse foreground means we only found ink.
    if cv2.contourArea(c) < 0.55 * bw * bh:
        return full_mask, full
    # An isolated leaf has scanner background on every side. A 'page' touching two or more image
    # borders is almost certainly a lighting gradient / cropped sheet -> use the whole frame (safe).
    touches = (x <= r) + (y <= r) + (x + bw >= w - r) + (y + bh >= h - r)
    if touches >= 2:
        return full_mask, full
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)   # fills holes inside the leaf
    return mask, (x, y, x + bw, y + bh)


# --------------------------------------------------------------------------- ink
@dataclass
class InkResult:
    ink: np.ndarray            # uint8 0/255 cleaned handwriting mask
    diff: np.ndarray           # uint8 "darkness vs. local background" map (contrast)
    holes: List[Box]           # binding-hole boxes (working resolution)
    page_mask: np.ndarray
    page_bbox: Box
    glyph_h: float             # median glyph height (px)


def _remove_long_lines(ink: np.ndarray, page_bbox: Box, cfg: Config) -> np.ndarray:
    """Delete ruling lines / page borders: very long thin horizontal or vertical structures."""
    pw = page_bbox[2] - page_bbox[0]
    ph = page_bbox[3] - page_bbox[1]
    lh = max(15, int(cfg.rule_line_frac_h * pw))
    lv = max(15, int(cfg.rule_line_frac_v * ph))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lh, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def extract_ink(bgr: np.ndarray, cfg: Config) -> InkResult:
    h, w = bgr.shape[:2]
    page_mask, page_bbox = detect_page_mask(bgr, cfg)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Replace the scanning background by the page's median brightness so it does not bias the
    # local background estimate near the leaf edge.
    page_med = int(np.median(gray[page_mask > 0]))
    g_fill = np.where(page_mask > 0, gray, page_med).astype(np.uint8)
    bg = estimate_background(g_fill, cfg.bg_median_kernel_frac)
    diff = np.clip(bg.astype(np.int16) - g_fill.astype(np.int16), 0, 255).astype(np.uint8)

    # Shrink the page mask: torn edges, shadows and leaf borders are not text.
    inset = max(2, int(cfg.page_inset_frac * min(page_bbox[2] - page_bbox[0], page_bbox[3] - page_bbox[1])))
    inner = cv2.erode(page_mask, np.ones((2 * inset + 1, 2 * inset + 1), np.uint8))
    # The hole (if any) must stay visible: erosion of a *filled* contour keeps it.
    vals = diff[inner > 0]
    if vals.size == 0:
        return InkResult(np.zeros((h, w), np.uint8), diff, [], page_mask, page_bbox, 10.0)
    otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu), float(cfg.min_ink_diff))
    ink = ((diff > thr) & (inner > 0)).astype(np.uint8) * 255

    # --- stains / binding holes: large, solid blobs -----------------------------------------
    holes: List[Box] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    page_area = float((page_bbox[2] - page_bbox[0]) * (page_bbox[3] - page_bbox[1]))
    remove = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        fill = a / float(bw * bh)
        aspect = bw / float(bh)
        if a < cfg.min_component_area:
            remove[i] = True
            continue
        if a / page_area >= cfg.stain_min_area_frac and fill >= cfg.stain_fill_ratio:
            remove[i] = True
            lo, hi = cfg.hole_area_frac
            if (cfg.detect_holes and lo <= a / page_area <= hi
                    and cfg.hole_fill_range[0] <= fill <= cfg.hole_fill_range[1]
                    and cfg.hole_aspect_range[0] <= aspect <= cfg.hole_aspect_range[1]):
                holes.append((int(x), int(y), int(x + bw), int(y + bh)))
    ink[remove[labels]] = 0

    ink = _remove_long_lines(ink, page_bbox, cfg)

    # re-clean tiny specks left after line removal
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < cfg.min_component_area
    small[0] = False
    ink[small[labels]] = 0

    heights = stats[1:, cv2.CC_STAT_HEIGHT][stats[1:, cv2.CC_STAT_AREA] >= cfg.min_component_area]
    glyph_h = float(np.median(heights)) if heights.size else 10.0
    glyph_h = float(np.clip(glyph_h, 4.0, 0.1 * h))
    return InkResult(ink, diff, holes, page_mask, page_bbox, glyph_h)


# --------------------------------------------------------------------------- skew
def estimate_skew(ink: np.ndarray, max_deg: float, step: float = 0.5) -> float:
    """Projection-profile skew estimate: the angle whose row-sum profile is 'peakiest'."""
    h, w = ink.shape
    s = 500.0 / max(h, w)
    small = cv2.resize(ink, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    if cv2.countNonZero(small) < 30:
        return 0.0
    ch, cw = small.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-6, step):
        m = cv2.getRotationMatrix2D((cw / 2, ch / 2), float(a), 1.0)
        rot = cv2.warpAffine(small, m, (cw, ch), flags=cv2.INTER_LINEAR)
        prof = rot.sum(axis=1, dtype=np.float64)
        score = float(np.var(prof))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def rotation_matrix(shape: Tuple[int, int], angle: float) -> np.ndarray:
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)


def median_glyph_height(ink: np.ndarray, min_area: int, max_h: float) -> float:
    """Median connected-component height = robust estimate of character size (script independent)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return 10.0
    sel = stats[1:, cv2.CC_STAT_AREA] >= min_area
    hs = stats[1:, cv2.CC_STAT_HEIGHT][sel]
    if hs.size == 0:
        return 10.0
    return float(np.clip(np.median(hs), 4.0, max_h))
