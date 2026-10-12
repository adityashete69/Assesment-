"""Draw predictions on a *copy* of the image."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .types import Region

# BGR colours, chosen to be distinct on both pale paper and dark palm leaf
COLORS = {
    "header": (0, 140, 255),     # orange
    "footer": (255, 0, 160),     # purple
    "main_text": (0, 170, 0),    # green
    "side_text": (255, 120, 0),  # blue
    "filler": (0, 0, 230),       # red
}


def annotate(bgr: np.ndarray, regions: Sequence[Region]) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    thick = max(2, int(round(min(h, w) / 300)))
    scale = max(0.5, min(h, w) / 900.0)
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        col = COLORS.get(r.label, (0, 0, 0))
        cv2.rectangle(out, (x0, y0), (x1, y1), col, thick)
        text = f"{r.label} {r.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick - 1))
        ty = y0 - 4 if y0 - th - base - 4 >= 0 else min(h - 1, y0 + th + 4)
        ty = max(th + 2, min(ty, h - base - 1))
        tx = max(0, min(x0, w - tw - 1))
        cv2.rectangle(out, (tx, ty - th - base // 2), (tx + tw, ty + base // 2), col, cv2.FILLED)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    return out
