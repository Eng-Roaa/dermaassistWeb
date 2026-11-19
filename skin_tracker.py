"""
skin_tracker.py  -  Skin Tracker logic (Day 2). No GUI code here, so it is testable anywhere.

* ROI geometry helpers (normalise / crop / convert mouse rectangle -> image pixels)
* TrackerService.save_visit(): writes the ORIGINAL image (lossless PNG, unmodified),
  the ROI crop, and the database row. Nothing is ever overwritten.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from skin_db import BASE, SkinDB

MIN_ROI_SIZE = 16  # pixels, in the original image


# ------------------------------------------------------------------ geometry
def normalize_roi(roi, img_w: int, img_h: int, min_size: int = MIN_ROI_SIZE):
    """Return (x, y, w, h) as ints clamped to the image. Raises ValueError if the
    rectangle is empty or smaller than `min_size` after clamping."""
    x, y, w, h = roi
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + h)))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        raise ValueError(f"The selected area is too small (minimum {min_size} px).")
    return (x0, y0, x1 - x0, y1 - y0)


def crop_roi(array: np.ndarray, roi) -> np.ndarray:
    """Crop an HxWxC numpy image with (x, y, w, h)."""
    x, y, w, h = roi
    return array[y:y + h, x:x + w]


def view_rect_to_image_roi(rect, scale: float, offset, img_w: int, img_h: int):
    """Convert a rectangle dragged on a canvas into original-image pixels.

    rect   = (x0, y0, x1, y1) canvas coordinates, any drag direction
    scale  = displayed size / original size
    offset = (ox, oy) canvas position of the image's top-left corner
    """
    x0, y0, x1, y1 = rect
    ox, oy = offset
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    ix0 = (left - ox) / scale
    iy0 = (top - oy) / scale
    ix1 = (right - ox) / scale
    iy1 = (bottom - oy) / scale
    return normalize_roi((ix0, iy0, ix1 - ix0, iy1 - iy0), img_w, img_h)


# ------------------------------------------------------------------- service
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class TrackerService:
    def __init__(self, db: SkinDB, base: Path = BASE):
        self.db = db
        self.base = Path(base)

    def save_visit(
        self,
        patient: dict,
        skin_region: str,
        image: Image.Image,
        roi=None,
        notes: str | None = None,
    ) -> dict:
        """Save one tracking visit and return its database row.

        Files:  images/<PatientID>/tracking/<PatientID>_<time>_<region>_original.png
                images/<PatientID>/tracking/<PatientID>_<time>_<region>_roi.png
        If anything fails, files written by this call are removed again.
        """
        image = image.convert("RGB")
        width, height = image.size
        if roi is not None:
            roi = normalize_roi(roi, width, height)

        folder = self.base / "images" / patient["patient_code"] / "tracking"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{patient['patient_code']}_{stamp}_{_slug(skin_region) or 'region'}"
        original_path = folder / f"{stem}_original.png"
        roi_path = folder / f"{stem}_roi.png" if roi else None

        written: list[Path] = []
        try:
            for path in (original_path, roi_path):
                if path is not None and path.exists():
                    raise FileExistsError(f"Refusing to overwrite {path.name}")
            image.save(original_path)               # PNG = lossless, unmodified pixels
            written.append(original_path)
            if roi:
                Image.fromarray(crop_roi(np.asarray(image), roi)).save(roi_path)
                written.append(roi_path)

            return self.db.add_visit(
                patient_id=patient["id"],
                skin_region=skin_region,
                original_image_path=original_path.relative_to(self.base).as_posix(),
                image_size=(width, height),
                roi=roi,
                roi_image_path=(roi_path.relative_to(self.base).as_posix()
                                if roi_path else None),
                notes=notes,
            )
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------- helpers
    def abs_path(self, relative: str) -> Path:
        return self.base / relative

    def roi_of(self, visit: dict):
        """(x, y, w, h) of a visit, or None if it has no ROI."""
        if visit.get("roi_w"):
            return (visit["roi_x"], visit["roi_y"], visit["roi_w"], visit["roi_h"])
        return None
