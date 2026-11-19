"""
demo_seed.py  -  demo accounts, synthetic demo patients, and the thermal SIMULATION.

Everything created here is synthetic. Patients carry the marker "SYNTHETIC DEMO PATIENT", which
is also what allows the FHIR "send" button to work (real patients can never be sent).

If the folder demo_assets/ contains  before.jpg  and  after.jpg  (photos of a consenting team
member, no face or tattoo), they are used for the demo patient instead of generated images.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fhir_export import SYNTHETIC_MARKER
from skin_db import SkinDB
from web_security import SecurityStore, Vault, sanitize_image


# ------------------------------------------------------------------ synthetic images
def synthetic_skin(seed: int, lesion: float = 0.0, shift=(0, 0), brightness: float = 0.0,
                   size=(600, 800)) -> Image.Image:
    """A skin-toned texture with an optional red patch. ILLUSTRATION ONLY, not a real photo."""
    rng = np.random.default_rng(seed)
    h, w = size

    def field(sigma):                                   # unit-variance smooth noise
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        return ((n - n.mean()) / n.std()).reshape(h, w, 1)

    base = np.zeros((h, w, 3), np.float32)
    base[:] = (222, 176, 150)                           # RGB skin tone
    img = base + field(40) * 14 + field(10) * 8 + field(2.2) * 5
    if lesion > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt((xx - w * 0.55) ** 2 + (yy - h * 0.5) ** 2)
        mask = np.clip(1 - d / (lesion * 140), 0, 1) ** 1.5
        img[..., 0] += mask * 18
        img[..., 1] -= mask * 70
        img[..., 2] -= mask * 55
    img = img + brightness
    if shift != (0, 0):
        img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ seeding
def seed_accounts(sec: SecurityStore, accounts: dict) -> int:
    """accounts = {username: {"password": ..., "role": "doctor"|"viewer", "name": optional}}"""
    n = 0
    for username, info in (accounts or {}).items():
        sec.upsert_user(username, info["password"], info.get("role", "viewer"), info.get("name"))
        n += 1
    return n


def _add_demo_visit(db: SkinDB, vault: Vault, patient_id: int, region: str, img: Image.Image,
                    days_ago: int, roi, notes: str):
    from skin_tracker import crop_roi, normalize_roi
    img, png = sanitize_image(_png(img))
    roi = normalize_roi(roi, *img.size)
    original = vault.put(png)
    roi_rel = vault.put(_png(Image.fromarray(crop_roi(np.asarray(img), roi))))
    when = (date.today() - timedelta(days=days_ago)).isoformat() + " 10:00:00"
    return db.add_visit(patient_id, region, original, img.size, roi=roi, roi_image_path=roi_rel,
                        notes=notes, visit_date=when)


def seed_demo_patients(db: SkinDB, vault: Vault, assets_dir: Path | None = None) -> bool:
    """Create the demo patients once. Returns True if anything was created."""
    if db.search_patients():
        return False
    note = f"{SYNTHETIC_MARKER}. No real person. Images are illustrations."
    a = db.add_patient("Demo Patient A", "1990-01-01", "Female", None, note)
    b = db.add_patient("Demo Patient B", "1985-06-15", "Male", None, note)

    assets = Path(assets_dir) if assets_dir else None
    real_before = assets / "before.jpg" if assets else None
    real_after = assets / "after.jpg" if assets else None
    use_real = bool(real_before and real_after and real_before.exists() and real_after.exists())

    if use_real:
        series = [(Image.open(real_before), 28, "baseline"), (Image.open(real_after), 0, "follow-up")]
    else:
        series = [(synthetic_skin(7, lesion=1.6), 28, "baseline (synthetic illustration)"),
                  (synthetic_skin(7, lesion=1.2, shift=(9, 5), brightness=10), 14, "week 2 (synthetic illustration)"),
                  (synthetic_skin(7, lesion=0.8, shift=(-6, 4), brightness=-6), 0, "week 4 (synthetic illustration)")]
    for img, ago, text in series:
        w, h = img.size
        _add_demo_visit(db, vault, a["id"], "Left Cheek", img, ago,
                        (int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.55)), text)
    db.add_treatment(a["id"], "Topical cream (demo)", (date.today() - timedelta(days=28)).isoformat(),
                     "Synthetic demo entry")
    img_b = synthetic_skin(21, lesion=0.9)
    w, h = img_b.size
    _add_demo_visit(db, vault, b["id"], "Right Arm", img_b, 3, (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)),
                    "single visit (synthetic illustration)")
    return True


def sample_scan_image() -> bytes:
    """A built-in synthetic image for trying the scan without uploading anything."""
    return _png(synthetic_skin(99, lesion=1.0))


# ------------------------------------------------------------------ thermal SIMULATION
def simulate_thermal(days: int = 14, seed: int = 5, rise_from_day: int = 9, rise_c: float = 2.4):
    """Synthetic daily skin-temperature readings for ONE body region.

    SIMULATED DATA. No sensor exists. This only illustrates how a future thermal module would
    present a patient's own baseline and a deviation from it. The threshold is a placeholder,
    not a clinically validated value.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    day_index = np.arange(days)
    temp = 33.1 + rng.normal(0, 0.15, days)
    temp[rise_from_day:] += np.linspace(0.6, rise_c, days - rise_from_day)
    baseline = float(np.mean(temp[:7]))
    df = pd.DataFrame({"day": day_index + 1, "temperature_c": np.round(temp, 2),
                       "own_baseline_c": round(baseline, 2)})
    df["deviation_c"] = (df["temperature_c"] - df["own_baseline_c"]).round(2)
    return df


def flag_deviation(df, threshold_c: float = 1.5):
    """Days where the simulated reading is above the patient's own baseline by more than the threshold."""
    return df[df["deviation_c"] > threshold_c]
