"""
image_comparison.py  -  Before / After comparison with OpenCV (Day 2)

compare_images():
  1. crop each image to its ROI (or use the whole image)
  2. resize both to a common working size
  3. try to align the AFTER image to the BEFORE image (ECC, affine)
  4. match the brightness (L channel only) so exposure changes matter less
  5. build a difference map (colour distance in LAB space)

IMPORTANT: the result is a PIXEL difference. It is not a medical measurement of
improvement or worsening. Lighting, angle, distance and skin tone all affect it.
All images are BGR uint8 arrays (OpenCV convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from skin_tracker import crop_roi

WORK_MAX_SIDE = 640        # working resolution (keeps the Raspberry Pi fast)
DELTA_E_FULL_SCALE = 30.0  # colour distance that maps to the hottest colour
CHANGE_THRESHOLD = 10.0    # colour distance counted as "changed"
ASPECT_WARN_RATIO = 0.20   # warn if the two ROIs differ >20% in aspect ratio


@dataclass
class ComparisonResult:
    before: np.ndarray                  # BGR crop, working size
    after: np.ndarray                   # BGR crop, aligned to `before`
    diff_map: np.ndarray                # BGR heat-map
    mean_delta_e: float                 # mean colour distance over valid pixels
    changed_percent: float              # % of pixels above CHANGE_THRESHOLD
    aligned: bool
    alignment_score: float | None       # ECC correlation (closer to 1 = better)
    warnings: list = field(default_factory=list)


# ------------------------------------------------------------------ helpers
def _fit_working_size(shape, max_side=WORK_MAX_SIDE):
    h, w = shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _align_ecc(before_bgr, after_bgr):
    """Return (aligned_after, valid_mask, score) or (None, None, None) on failure."""
    h, w = before_bgr.shape[:2]
    ref = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    mov = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        score, warp = cv2.findTransformECC(
            ref, mov, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return None, None, None

    # Reject absurd transforms (the optimiser sometimes "succeeds" on noise).
    linear = warp[:, :2]
    det = float(np.linalg.det(linear))
    shift = float(np.hypot(warp[0, 2] / w, warp[1, 2] / h))
    if not (0.7 < det < 1.4) or shift > 0.30 or score < 0.3:
        return None, None, None

    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(after_bgr, warp, (w, h), flags=flags,
                             borderMode=cv2.BORDER_REPLICATE)
    ones = np.full((h, w), 255, dtype=np.uint8)
    mask = cv2.warpAffine(ones, warp, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, mask > 200, float(score)


def _to_lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _match_luminance(lab_before, lab_after, valid):
    """Give the AFTER image the same mean/std of lightness as BEFORE.
    Only L is touched, so real colour changes (e.g. redness) are preserved."""
    lb = lab_before[..., 0][valid]
    la = lab_after[..., 0][valid]
    if lb.size == 0 or la.std() < 1e-3:
        return lab_after
    out = lab_after.copy()
    out[..., 0] = np.clip(
        (lab_after[..., 0] - la.mean()) / la.std() * lb.std() + lb.mean(), 0, 100)
    return out


# --------------------------------------------------------------------- main
def compare_images(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    before_roi=None,
    after_roi=None,
    normalize_lighting: bool = True,
    try_align: bool = True,
) -> ComparisonResult:
    warnings: list[str] = []

    before = crop_roi(before_bgr, before_roi) if before_roi else before_bgr
    after = crop_roi(after_bgr, after_roi) if after_roi else after_bgr
    if before.size == 0 or after.size == 0:
        raise ValueError("One of the images/regions is empty.")

    ratio_b = before.shape[1] / before.shape[0]
    ratio_a = after.shape[1] / after.shape[0]
    if abs(ratio_a - ratio_b) / ratio_b > ASPECT_WARN_RATIO:
        warnings.append("The two selected areas have different shapes; "
                        "the later one was stretched to match.")

    size = _fit_working_size(before.shape)
    before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
    after = cv2.resize(after, size, interpolation=cv2.INTER_AREA)

    valid = np.ones(before.shape[:2], dtype=bool)
    aligned, score = False, None
    if try_align:
        warped, mask, score = _align_ecc(before, after)
        if warped is not None:
            after, valid, aligned = warped, mask, True
        else:
            warnings.append("Automatic alignment failed; images were compared "
                            "without alignment (shifts will look like changes).")

    lab_b = _to_lab(cv2.GaussianBlur(before, (5, 5), 0))
    lab_a = _to_lab(cv2.GaussianBlur(after, (5, 5), 0))
    if normalize_lighting:
        lab_a = _match_luminance(lab_b, lab_a, valid)

    delta_e = np.sqrt(((lab_b - lab_a) ** 2).sum(axis=2))
    scaled = np.clip(delta_e / DELTA_E_FULL_SCALE * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    heat[~valid] = 0

    if valid.any():
        mean_de = float(delta_e[valid].mean())
        changed = float((delta_e[valid] > CHANGE_THRESHOLD).mean() * 100.0)
    else:
        mean_de, changed = 0.0, 0.0
        warnings.append("No overlapping area after alignment.")

    return ComparisonResult(before, after, heat, mean_de, changed,
                            aligned, score, warnings)


def make_comparison_sheet(result: ComparisonResult, before_label: str,
                          after_label: str) -> np.ndarray:
    """One image: BEFORE | AFTER (aligned) | DIFFERENCE MAP, with a caption bar."""
    h, w = result.before.shape[:2]
    bar = 34
    panels = [(result.before, before_label),
              (result.after, after_label),
              (result.diff_map, "Difference map")]
    tiles = []
    for img, label in panels:
        tile = np.full((h + bar, w, 3), 30, dtype=np.uint8)
        tile[bar:, :] = img
        cv2.putText(tile, label[:42], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    gap = np.full((h + bar, 6, 3), 30, dtype=np.uint8)
    return np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]])
