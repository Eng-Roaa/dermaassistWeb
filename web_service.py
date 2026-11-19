"""
web_service.py  -  everything the web UI can do, with permission checks and audit logging.

No Streamlit code here, so it is fully testable. The UI (app.py) only calls these methods.
Images are always stored encrypted (see web_security.Vault).
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from clinical_summary import build_timeline, patient_summary
from fhir_export import SYNTHETIC_MARKER
from image_comparison import compare_images, make_comparison_sheet
from image_quality import assess_quality
from skin_db import SKIN_REGIONS, SkinDB
from skin_tracker import crop_roi, normalize_roi
from web_security import SecurityStore, Vault, can, sanitize_image

DISCLAIMER = ("AI-assisted PRELIMINARY classification. Not a diagnosis. "
              "The qualified clinician makes every diagnostic, treatment and referral decision.")
_MODEL_LOCK = threading.Lock()         # the TFLite interpreter must not run two images at once


def load_classifier(model_path: Path):
    """Return (classifier, reason). classifier is None if the model or runtime is missing."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None, f"Model file not found: {model_path.name}"
    try:
        from skin_classifier import SkinClassifier
        return SkinClassifier(model_path), ""
    except Exception as exc:                          # missing runtime, bad file, ...
        return None, f"Classifier unavailable: {exc}"


class WebService:
    def __init__(self, db: SkinDB, sec: SecurityStore, vault: Vault, classifier=None):
        self.db, self.sec, self.vault, self.classifier = db, sec, vault, classifier

    # ----------------------------------------------------------------- helpers
    def _need(self, user: dict, permission: str):
        if not user or not can(user.get("role"), permission):
            raise PermissionError("Your role does not allow this action.")

    def _audit(self, user: dict, action: str, target="", detail=""):
        self.sec.log(user["username"], user["role"], action, target, detail)

    # ----------------------------------------------------------------- reads
    def list_patients(self, user: dict, query: str = "") -> list[dict]:
        self._need(user, "read")
        return self.db.search_patients(query)

    def open_patient(self, user: dict, patient_id: int) -> dict:
        self._need(user, "read")
        summary = patient_summary(self.db, patient_id)
        summary["timeline"] = build_timeline(summary["visits"], summary["treatments"])
        summary["reviews"] = self.sec.reviews(patient_id)
        self._audit(user, "view_patient", summary["patient"]["patient_code"])
        return summary

    def visit_image(self, visit: dict, kind: str = "original") -> Image.Image:
        rel = visit["roi_image_path"] if kind == "roi" and visit["roi_image_path"] else visit["original_image_path"]
        return Image.open(io.BytesIO(self.vault.get(rel))).convert("RGB")

    def _compare(self, first: dict, second: dict):
        """Comparison without permission check or audit entry (callers do both)."""
        imgs = []
        for v in (first, second):
            data = np.frombuffer(self.vault.get(v["original_image_path"]), dtype=np.uint8)
            imgs.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
        roi = lambda v: (v["roi_x"], v["roi_y"], v["roi_w"], v["roi_h"]) if v.get("roi_w") else None
        result = compare_images(imgs[0], imgs[1], roi(first), roi(second))
        sheet = make_comparison_sheet(result, f"V{first['visit_number']} {first['visit_date'][:10]}",
                                      f"V{second['visit_number']} {second['visit_date'][:10]}")
        return result, sheet[:, :, ::-1].copy()

    def compare(self, user: dict, first: dict, second: dict):
        """Returns (ComparisonResult, sheet as RGB array)."""
        self._need(user, "read")
        result, sheet = self._compare(first, second)
        self._audit(user, "compare_visits", first["patient_id"],
                    f"V{first['visit_number']} vs V{second['visit_number']}")
        return result, sheet

    def image_quality(self, image_bytes: bytes) -> dict:
        """Brightness / contrast / detail indicators for an upload. Nothing is stored."""
        img, _ = sanitize_image(image_bytes)
        return assess_quality(img)

    def patient_report(self, user: dict, patient_id: int) -> bytes:
        """The one-patient PDF report (read-only users may export it too)."""
        self._need(user, "export")
        from report_pdf import generate_patient_report      # reportlab is imported only when needed
        pdf = generate_patient_report(self, patient_id)
        self._audit(user, "export_pdf_report", f"patient {patient_id}")
        return pdf

    # ----------------------------------------------------------------- writes
    def add_patient(self, user: dict, full_name: str, date_of_birth: str | None = None,
                    sex: str | None = None, phone: str | None = None, notes: str | None = None,
                    confirm_synthetic: bool = False) -> dict:
        """Register a patient. This proof of concept accepts synthetic / consented demo data only,
        so the clinician must confirm it, and the record is marked accordingly."""
        self._need(user, "write")
        if not confirm_synthetic:
            raise ValueError("Please confirm that this is a synthetic or consenting-volunteer record, "
                             "not a real identifiable patient.")
        notes = (notes or "").strip()
        notes = f"{SYNTHETIC_MARKER}. {notes}" if notes else f"{SYNTHETIC_MARKER}."
        patient = self.db.add_patient(full_name, date_of_birth, sex, phone, notes)
        self._audit(user, "add_patient", patient["patient_code"])
        return patient

    def add_visit(self, user: dict, patient_id: int, region: str, image_bytes: bytes,
                  roi=None, notes: str | None = None) -> dict:
        self._need(user, "write")
        if region not in SKIN_REGIONS:
            raise ValueError("Choose a skin region.")
        img, png = sanitize_image(image_bytes)
        width, height = img.size
        roi_values = normalize_roi(roi, width, height) if roi else None
        original_rel = self.vault.put(png)
        roi_rel = None
        if roi_values:
            buf = io.BytesIO()
            Image.fromarray(crop_roi(np.asarray(img), roi_values)).save(buf, "PNG")
            roi_rel = self.vault.put(buf.getvalue())
        visit = self.db.add_visit(patient_id, region, original_rel, (width, height),
                                  roi=roi_values, roi_image_path=roi_rel, notes=notes)
        self._audit(user, "add_visit", visit["patient_id"], f"visit #{visit['visit_number']} {region}")
        return visit

    def classify(self, user: dict, image_bytes: bytes) -> dict:
        """Run the classifier on an uploaded image. Nothing is stored here."""
        self._need(user, "read")
        if self.classifier is None:
            raise RuntimeError("The classifier is not available on this server.")
        img, _ = sanitize_image(image_bytes)
        with _MODEL_LOCK:
            pred = self.classifier.predict(img)
        self._audit(user, "ai_scan", "", f"{'uncertain' if pred.uncertain else pred.major_class}")
        return {
            "model": pred.model_name, "top_class": pred.major_class, "confidence": pred.confidence,
            "uncertain": pred.uncertain, "scores": pred.scores,
            "processing_time": pred.processing_time, "disclaimer": DISCLAIMER,
        }

    def save_scan(self, user: dict, patient_id: int, image_bytes: bytes, prediction: dict) -> int:
        """Store an analysed image (encrypted) and its result in the patient's record."""
        self._need(user, "write")
        _, png = sanitize_image(image_bytes)
        rel = self.vault.put(png)
        capture_id = self.db.add_capture(patient_id, rel)
        self.db.add_prediction(capture_id, prediction["model"], prediction["top_class"],
                               prediction["confidence"], prediction["uncertain"],
                               prediction["scores"], prediction["processing_time"])
        self._audit(user, "save_scan", f"patient {patient_id}", f"capture {capture_id}")
        return capture_id

    def review(self, user: dict, patient_id: int, capture_id: int | None, ai_agreement: str,
               action: str, note: str = "") -> int:
        return self.sec.add_review(user, patient_id, capture_id, ai_agreement, action, note)

    def add_treatment(self, user: dict, patient_id: int, name: str, start: str | None,
                      notes: str | None = None, visit_id: int | None = None) -> dict:
        self._need(user, "write")
        t = self.db.add_treatment(patient_id, name, start, notes, visit_id)
        self._audit(user, "add_treatment", f"patient {patient_id}", t["treatment_name"])
        return t

    def stop_treatment(self, user: dict, treatment_id: int) -> dict:
        self._need(user, "write")
        t = self.db.stop_treatment(treatment_id)
        self._audit(user, "stop_treatment", f"treatment {treatment_id}")
        return t

    def audit_events(self, user: dict, limit: int = 200) -> list[dict]:
        self._need(user, "audit")
        return self.sec.audit_events(limit)
