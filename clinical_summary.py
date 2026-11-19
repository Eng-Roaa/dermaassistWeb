"""
clinical_summary.py  -  data for the Doctor Dashboard and the PDF report (Day 3).

Pure logic (no GUI, no OpenCV): everything here is testable on any machine.
"""

from __future__ import annotations

from collections import Counter

from skin_db import SkinDB, age_from_dob

# Order on the same day: the visit is examined first, then treatment starts/ends.
_RANK = {"visit": 0, "treatment_start": 1, "treatment_end": 2}


def treatments_active_on(treatments: list[dict], day: str) -> list[dict]:
    """Treatments in progress on `day` (YYYY-MM-DD, or a timestamp: first 10 chars used)."""
    day = day[:10]
    return [t for t in treatments
            if t["start_date"] <= day and (t["end_date"] is None or day <= t["end_date"])]


def patient_summary(db: SkinDB, patient_id: int) -> dict:
    """Everything the dashboard shows for one patient."""
    patient = db.get_patient(patient_id)
    if patient is None:
        raise ValueError("Patient not found.")

    scans = db.get_patient_history(patient_id)           # newest first
    analysed = [s for s in scans if s["predicted_class"] is not None]
    confident = [s for s in analysed if not s["is_uncertain"]]
    class_counts = Counter(s["predicted_class"] for s in confident)

    visits = db.list_visits(patient_id)                  # oldest first
    treatments = db.list_treatments(patient_id)

    return {
        "patient": patient,
        "age": age_from_dob(patient["date_of_birth"]),
        "scan_count": len(scans),
        "analysed_count": len(analysed),
        "uncertain_count": len(analysed) - len(confident),
        "class_counts": dict(class_counts.most_common()),
        "latest_scan": analysed[0] if analysed else None,
        "visit_count": len(visits),
        "first_visit": visits[0] if visits else None,
        "last_visit": visits[-1] if visits else None,
        "visit_dates": [v["visit_date"] for v in visits],
        "region_counts": dict(Counter(v["skin_region"] for v in visits).most_common()),
        "treatment_count": len(treatments),
        "active_treatments": [t for t in treatments if t["end_date"] is None],
        "treatments": treatments,
        "visits": visits,
        "scans": scans,
    }


def build_timeline(visits: list[dict], treatments: list[dict]) -> list[dict]:
    """One chronological list mixing tracking visits and treatment events, so the
    doctor can read visual follow-up next to what was being treated.

    Each row: {date, kind, text, ref_id}. For a visit row the text also lists the
    treatments that were in progress on that day."""
    rows = []
    for v in visits:
        active = treatments_active_on(treatments, v["visit_date"])
        text = f"Visit #{v['visit_number']} - {v['skin_region']}"
        if active:
            text += "  |  on: " + ", ".join(t["treatment_name"] for t in active)
        rows.append({"date": v["visit_date"][:10], "kind": "visit", "text": text,
                     "ref_id": v["id"], "_order": (v["visit_date"][:10], _RANK["visit"],
                                                   v["visit_number"])})
    for t in treatments:
        rows.append({"date": t["start_date"], "kind": "treatment_start",
                     "text": f"Treatment started: {t['treatment_name']}",
                     "ref_id": t["id"], "_order": (t["start_date"],
                                                   _RANK["treatment_start"], t["id"])})
        if t["end_date"]:
            rows.append({"date": t["end_date"], "kind": "treatment_end",
                         "text": f"Treatment stopped: {t['treatment_name']}",
                         "ref_id": t["id"], "_order": (t["end_date"],
                                                       _RANK["treatment_end"], t["id"])})
    rows.sort(key=lambda r: r["_order"])
    for r in rows:
        del r["_order"]
    return rows


def first_last_by_region(visits: list[dict]) -> dict:
    """{region: (first_visit, latest_visit)} for regions with at least two visits."""
    by_region: dict[str, list[dict]] = {}
    for v in visits:
        by_region.setdefault(v["skin_region"], []).append(v)
    return {region: (items[0], items[-1])
            for region, items in by_region.items() if len(items) >= 2}
