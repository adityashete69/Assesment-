#!/usr/bin/env python3
"""Batch inference CLI.

    python inference.py --input ./data/samples --output ./results

Writes, for every image:  <output>/annotated/<name>_annotated.png  and  <output>/json/<name>.json
and a combined <output>/predictions.json. Input files are opened read-only and never modified.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2

from layout_detector import Config, LayoutDetector
from layout_detector.io_utils import list_images, load_image_bgr, write_json
from layout_detector.visualize import annotate

log = logging.getLogger("inference")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect header / footer / main_text / side_text / filler regions "
                                            "in manuscript images.")
    p.add_argument("--input", required=True, help="image file or folder of images")
    p.add_argument("--output", required=True, help="output folder (created if missing)")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    p.add_argument("--no-ocr", action="store_true", help="skip English-text (filler) OCR")
    p.add_argument("--no-deskew", action="store_true", help="disable skew correction during analysis")
    p.add_argument("--work-size", type=int, default=Config.work_max_side,
                   help="analysis resolution (long side, px). Larger = slower, finer")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    in_path, out_dir = Path(args.input), Path(args.output)
    if not in_path.exists():
        log.error("Input path does not exist: %s", in_path)
        return 2
    if in_path.is_file() and in_path.resolve() == out_dir.resolve():
        log.error("--output must be a folder, not the input file")
        return 2

    images = list_images(in_path, recursive=not args.no_recursive)
    # never re-process our own outputs if --output lives inside --input
    images = [p for p in images if not _is_inside(p, out_dir)]
    if not images:
        log.error("No supported images found in %s", in_path)
        return 2

    cfg = Config(use_ocr=not args.no_ocr, deskew=not args.no_deskew, work_max_side=args.work_size)
    detector = LayoutDetector(cfg)
    base = in_path if in_path.is_dir() else in_path.parent
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)

    summary, errors = [], []
    t_all = time.time()
    for i, path in enumerate(images, 1):
        rel = path.relative_to(base)
        stem = "__".join(rel.with_suffix("").parts)      # keeps sub-folder images from colliding
        try:
            t0 = time.time()
            bgr = load_image_bgr(path)
            result = detector.detect(bgr)
            record = result.to_dict(str(rel).replace("\\", "/"))
            write_json(record, out_dir / "json" / f"{stem}.json")
            cv2.imwrite(str(out_dir / "annotated" / f"{stem}_annotated.png"), annotate(bgr, result.regions))
            summary.append(record)
            log.info("[%d/%d] %s -> %d regions (%.1fs)", i, len(images), rel, len(result.regions), time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            log.exception("Failed on %s", path)
            errors.append({"image": str(rel), "error": f"{type(exc).__name__}: {exc}"})

    write_json({"num_images": len(images), "num_failed": len(errors),
                "results": summary, "errors": errors}, out_dir / "predictions.json")
    log.info("Done: %d ok, %d failed in %.1fs -> %s", len(summary), len(errors), time.time() - t_all, out_dir)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(run())
