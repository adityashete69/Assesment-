"""All tunable parameters live here so behaviour is reproducible and documented."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- working resolution -------------------------------------------------
    work_max_side: int = 1600          # analysis runs on a copy whose long side <= this

    # --- page detection -----------------------------------------------------
    page_min_area_frac: float = 0.15   # smaller "pages" are rejected -> whole image
    page_inset_frac: float = 0.015     # ignore ink this close to the page edge (shadows, torn edges)

    # --- ink extraction -----------------------------------------------------
    bg_median_kernel_frac: float = 0.06  # background estimate window (fraction of long side)
    min_ink_diff: int = 18               # floor for the ink threshold (faded ink friendly)
    min_component_area: int = 6          # px at working resolution
    rule_line_frac_h: float = 0.88       # horizontal ruling lines: near full page width (headline strokes are shorter)
    rule_line_frac_v: float = 0.45       # vertical ruling lines: longer than this * page height
    stain_fill_ratio: float = 0.55       # big, solid blobs are stains/holes, not handwriting
    stain_min_area_frac: float = 0.0004

    # --- skew ---------------------------------------------------------------
    deskew: bool = True
    max_skew_deg: float = 5.0
    min_skew_deg: float = 0.5

    # --- block segmentation -------------------------------------------------
    block_kx: float = 2.0   # horizontal merge distance in units of median glyph height
    block_ky: float = 0.8   # vertical merge distance in units of median glyph height
    min_block_ink_frac: float = 0.003  # blocks with < this share of the largest block's ink are noise
    min_block_ink_px: int = 40     # absolute floor; also >= 0.4 * glyph_h^2

    # --- classification -----------------------------------------------------
    header_band_frac: float = 0.22     # header must start in the top 22% of the page
    footer_band_frac: float = 0.22
    side_band_frac: float = 0.28       # side_text must sit in the outer 28% of the page width
    second_column_ink_frac: float = 0.35  # a detached block this heavy is another column, not a note
    faint_contrast_ratio: float = 0.55  # detached blocks fainter than this vs body ink -> filler (pencil)

    # --- filler: English overlay text (OCR) --------------------------------
    use_ocr: bool = True
    ocr_lang: str = "eng"
    ocr_band_frac: float = 0.15        # scan the top/bottom 15% of the image
    ocr_min_conf: int = 70
    ocr_pad: int = 3

    # --- filler: binding holes ---------------------------------------------
    detect_holes: bool = True
    hole_area_frac: tuple = (0.0003, 0.012)
    hole_fill_range: tuple = (0.6, 0.9)    # circle fills ~0.785 of its bounding box
    hole_aspect_range: tuple = (0.7, 1.4)
