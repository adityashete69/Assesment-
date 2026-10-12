"""Small, dependency-free box helpers (all boxes are (x0, y0, x1, y1))."""
from __future__ import annotations

from typing import Iterable, Tuple

Box = Tuple[int, int, int, int]


def clip_box(box: Iterable[float], width: int, height: int) -> Box:
    """Clamp a box to [0, width-1] x [0, height-1], guaranteeing x1 > x0 and y1 > y0."""
    x0, y0, x1, y1 = box
    x0 = int(round(max(0, min(x0, width - 1))))
    x1 = int(round(max(0, min(x1, width - 1))))
    y0 = int(round(max(0, min(y0, height - 1))))
    y1 = int(round(max(0, min(y1, height - 1))))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
        x0 = max(0, x1 - 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)
        y0 = max(0, y1 - 1)
    return (x0, y0, x1, y1)


def scale_box(box: Iterable[float], factor: float) -> Tuple[float, float, float, float]:
    return tuple(v * factor for v in box)  # type: ignore[return-value]


def area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def intersection_area(a: Box, b: Box) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))
