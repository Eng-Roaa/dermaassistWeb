"""
image_quality.py  -  quick image-quality indicators for uploaded skin photos.

Ported from the Raspberry Pi project (image_quality.py). These are PRELIMINARY technical
indicators only (brightness, contrast, detail). They do NOT establish medical suitability.
"""

from __future__ import annotations

from PIL import Image, ImageFilter, ImageStat

DARK_BELOW = 50
BRIGHT_ABOVE = 220
LOW_CONTRAST_BELOW = 20
LOW_DETAIL_BELOW = 10
SMALL_SIDE = 300


def assess_quality(image: Image.Image) -> dict:
    """Return brightness / contrast / detail scores and a list of plain-language warnings."""
    img = image.convert("RGB")
    width, height = img.size
    probe = img.copy()
    probe.thumbnail((1000, 1000))                       # keeps the check fast on large photos
    gray = probe.convert("L")
    brightness = ImageStat.Stat(gray).mean[0]
    contrast = ImageStat.Stat(gray).stddev[0]
    detail = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).stddev[0]

    warnings = []
    if min(width, height) < SMALL_SIDE:
        warnings.append(f"The image is small ({width}x{height}px); fine detail may be lost.")
    if brightness < DARK_BELOW:
        warnings.append("The image may be too dark.")
    elif brightness > BRIGHT_ABOVE:
        warnings.append("The image may be overexposed.")
    if contrast < LOW_CONTRAST_BELOW:
        warnings.append("The image may have low contrast.")
    if detail < LOW_DETAIL_BELOW:
        warnings.append("The image may lack visible detail (blurred or out of focus).")
    return {"width": width, "height": height, "brightness": round(brightness, 1),
            "contrast": round(contrast, 1), "detail": round(detail, 1),
            "warnings": warnings, "ok": not warnings}
