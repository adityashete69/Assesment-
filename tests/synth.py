"""Seeded synthetic manuscript pages (random pen-stroke 'words'), so tests need no external data."""
from __future__ import annotations

import cv2
import numpy as np


def _word(canvas, x, y, width, height, rng, ink=(35, 35, 40), thick=3):
    """Draw a pseudo-word: a few random polylines in a (width x height) cell + a headline stroke."""
    cv2.line(canvas, (x, y + 4), (x + width, y + 4), ink, thick - 1, cv2.LINE_AA)
    n_glyphs = max(2, width // 18)
    for g in range(n_glyphs):
        gx = x + g * (width // n_glyphs)
        pts = [(gx + int(rng.integers(0, 16)), y + int(rng.integers(4, height))) for _ in range(int(rng.integers(3, 6)))]
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, ink, thick, cv2.LINE_AA)


def draw_line(canvas, x0, x1, y, rng, height=30, **kw):
    x = x0
    while x < x1 - 40:
        w = int(rng.integers(50, 130))
        w = min(w, x1 - x)
        _word(canvas, x, y, w, height, rng, **kw)
        x += w + int(rng.integers(12, 22))


def make_page(width=1400, height=1000, seed=0, header=True, footer=True, side=True,
              bg=(205, 222, 232), ink=(35, 35, 40), n_lines=9):
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), bg, np.uint8)
    img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
    body_x0, body_x1 = 300, width - 140
    top = 250
    expected = {"main_text": (body_x0, top, body_x1, top + (n_lines - 1) * 52 + 30)}
    for i in range(n_lines):
        draw_line(img, body_x0, body_x1, top + i * 52, rng, ink=ink)
    if header:
        draw_line(img, 560, 900, 90, rng, ink=ink)
        expected["header"] = (560, 90, 900, 120)
    if footer:
        draw_line(img, 650, 900, height - 110, rng, ink=ink)
        expected["footer"] = (650, height - 110, 900, height - 80)
    if side:
        for k in range(4):
            draw_line(img, 60, 190, top + 60 + k * 52, rng, height=26, ink=ink)
        expected["side_text"] = (60, top + 60, 190, top + 60 + 3 * 52 + 26)
    return img, expected


def degrade(img, blur=0.0, contrast=1.0, gradient=0.0, noise=0.0, seed=1):
    """Apply faded ink / uneven lighting / blur / sensor noise."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * contrast + mean                          # fade
    if gradient:
        h, w = out.shape[:2]
        ramp = np.linspace(1 - gradient, 1 + gradient * 0.3, w, dtype=np.float32)[None, :, None]
        out *= ramp                                               # uneven lighting
    if noise:
        out += rng.normal(0, noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out


def rotate(img, deg, border=(205, 222, 232)):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=border)
