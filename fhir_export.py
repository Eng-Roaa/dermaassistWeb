"""
fhir_export.py  -  HL7 FHIR R4 export (proof of concept)

build_bundle()  creates a FHIR R4 *transaction* Bundle for one patient:
    Patient, Encounter (one per tracking visit), Media (photo metadata, NOT the image),
    Observation (AI result, status "preliminary"), ClinicalImpression (clinician review),
    CarePlan (treatments)
Every resource has an identifier and a conditional create (ifNoneExist), so sending the
same bundle twice does not create duplicates.

send_bundle()   POSTs it to a FHIR server. Only patients marked as synthetic demo patients
                may be sent, and the default server is the PUBLIC HAPI test server, which
                must never receive real patient data.

Images are never exported: only metadata. The AI result is exported as a preliminary
observation, never as a diagnosis.
"""

from __future__ import annotations

import json
import uuid

import requests

from skin_db import SkinDB
from web_security import SecurityStore

IDENT_SYSTEM = "urn:dermaassist:demo"
DEFAULT_FHIR_BASE = "https://hapi.fhir.org/baseR4"
SYNTHETIC_MARKER = "SYNTHETIC DEMO PATIENT"
AI_NOTE = ("AI-assisted preliminary classification from an experimental model. "
           "Not a diagnosis; must be reviewed by a qualified clinician.")
_GENDER = {"Male": "male", "Female": "female", "Other": "other"}
_NS = uuid.UUID("6f1f3e0e-3b53-4c63-9c1e-1f2f4a0c9a11")


def is_synthetic(patient: dict) -> bool:
    return SYNTHETIC_MARKER in (patient.get("notes") or "")


def _uid(key: str) -> str:
    return "urn:uuid:" + str(uuid.uuid5(_NS, key))


def _entry(resource: dict, key: str) -> dict:
    ident = f"{IDENT_SYSTEM}|{key}"
    resource["identifier"] = [{"system": IDENT_SYSTEM, "value": key}]
    return {"fullUrl": _uid(key), "resource": resource,
            "request": {"method": "POST", "url": resource["resourceType"], "ifNoneExist": f"identifier={ident}"}}


def build_bundle(db: SkinDB, sec: SecurityStore, patient_id: int) -> dict:
    patient = db.get_patient(patient_id)
    if patient is None:
        raise ValueError("Patient not found.")
    code = patient["patient_code"]
    p_ref = {"reference": _uid(f"{code}")}
    entries = []

    p = {"resourceType": "Patient", "name": [{"text": patient["full_name"]}], "active": True}
    if patient.get("sex"):
        p["gender"] = _GENDER.get(patient["sex"], "unknown")
    if patient.get("date_of_birth"):
        p["birthDate"] = patient["date_of_birth"]
    if patient.get("phone"):
        p["telecom"] = [{"system": "phone", "value": patient["phone"]}]
    entries.append(_entry(p, code))

    # ---- tracking visits: Encounter + Media (metadata only)
    for v in db.list_visits(patient_id):
        vkey = f"{code}-visit-{v['visit_number']}"
        enc = {"resourceType": "Encounter", "status": "finished",
               "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB",
                         "display": "ambulatory"},
               "subject": p_ref, "period": {"start": v["visit_date"][:10]},
               "reasonCode": [{"text": f"Skin follow-up photography: {v['skin_region']}"}]}
        entries.append(_entry(enc, vkey))
        media = {"resourceType": "Media", "status": "completed",
                 "type": {"text": "Photo"}, "subject": p_ref, "encounter": {"reference": _uid(vkey)},
                 "createdDateTime": v["visit_date"][:10], "bodySite": {"text": v["skin_region"]},
                 "content": {"contentType": "image/png",
                             "title": "Original image kept encrypted on the platform (not exported)"}}
        if v.get("notes"):
            media["note"] = [{"text": v["notes"]}]
        entries.append(_entry(media, f"{vkey}-media"))

    # ---- AI results: preliminary observations, never diagnoses
    for s in db.get_patient_history(patient_id):
        if s["predicted_class"] is None:
            continue
        skey = f"{code}-scan-{s['capture_id']}"
        value = "Uncertain (low confidence)" if s["is_uncertain"] else s["predicted_class"]
        obs = {"resourceType": "Observation", "status": "preliminary",
               "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                         "code": "imaging", "display": "Imaging"}]}],
               "code": {"text": "AI-assisted preliminary skin image classification (not a diagnosis)"},
               "subject": p_ref, "effectiveDateTime": s["captured_at"][:10], "valueString": value,
               "method": {"text": f"Experimental classifier ({s['model_name']})"},
               "note": [{"text": AI_NOTE}],
               "component": [{"code": {"text": name},
                              "valueQuantity": {"value": round(score * 100, 2), "unit": "%",
                                                "system": "http://unitsofmeasure.org", "code": "%"}}
                             for name, score in (s["scores"] or {}).items()]}
        entries.append(_entry(obs, skey))

    # ---- clinician reviews
    for r in sec.reviews(patient_id):
        impression = {"resourceType": "ClinicalImpression", "status": "completed", "subject": p_ref,
                      "date": r["created_at"][:10], "assessor": {"display": r["username"]},
                      "summary": f"{r['ai_agreement']}. Decision: {r['action']}."}
        if r["note"]:
            impression["note"] = [{"text": r["note"]}]      # FHIR forbids empty arrays
        entries.append(_entry(impression, f"{code}-review-{r['id']}"))

    # ---- treatments
    for t in db.list_treatments(patient_id):
        plan = {"resourceType": "CarePlan", "status": "completed" if t["end_date"] else "active",
                "intent": "plan", "title": t["treatment_name"], "subject": p_ref,
                "category": [{"text": "Dermatology treatment"}],
                "period": {"start": t["start_date"], **({"end": t["end_date"]} if t["end_date"] else {})},
                "activity": [{"detail": {"description": t["treatment_name"],
                                         "status": "completed" if t["end_date"] else "in-progress"}}]}
        if t.get("notes"):
            plan["description"] = t["notes"]
        entries.append(_entry(plan, f"{code}-treatment-{t['id']}"))

    return {"resourceType": "Bundle", "type": "transaction",
            "meta": {"tag": [{"system": IDENT_SYSTEM, "code": "demo-export",
                              "display": "DermaAssist proof of concept export"}]},
            "entry": entries}


def bundle_json(bundle: dict) -> str:
    return json.dumps(bundle, indent=2, ensure_ascii=False)


def send_bundle(bundle: dict, patient: dict, base_url: str = DEFAULT_FHIR_BASE,
                timeout: int = 30, post=None) -> dict:
    """POST the bundle to a FHIR server. Refuses anything that is not a synthetic demo patient."""
    if not is_synthetic(patient):
        raise PermissionError("Only synthetic demo patients may be sent to a FHIR server.")
    if not base_url.startswith("https://"):
        raise ValueError("The FHIR server URL must use HTTPS.")
    response = (post or requests.post)(base_url, data=json.dumps(bundle),
                    headers={"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"},
                    timeout=timeout)
    ok = 200 <= response.status_code < 300
    return {"ok": ok, "status": response.status_code,
            "search_url": f"{base_url}/Patient?identifier={IDENT_SYSTEM}|{patient['patient_code']}",
            "detail": "" if ok else response.text[:500]}
