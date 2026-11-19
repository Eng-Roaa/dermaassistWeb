"""Run with:  python3 test_web.py     (temporary folders only; no network is used)"""

import io
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from demo_seed import (flag_deviation, sample_scan_image, seed_accounts, seed_demo_patients,
                       simulate_thermal, synthetic_skin)
from fhir_export import (IDENT_SYSTEM, SYNTHETIC_MARKER, build_bundle, bundle_json, is_synthetic,
                         send_bundle)
from skin_db import SkinDB
from web_security import (MAX_FAILURES, MAX_UPLOAD_BYTES, SecurityStore, Vault, can,
                          hash_password, sanitize_image, verify_password)
from web_service import WebService


def expect(exc_type, func, *a, **k):
    try:
        func(*a, **k)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} from {getattr(func, '__name__', func)}")


OPEN_DBS = []                      # closed at the end (Windows cannot delete a folder with an open database)


def make_system(tmp: Path, classifier=None):
    db = SkinDB(tmp / "database" / "t.db")
    OPEN_DBS.append(db)
    sec = SecurityStore(db.conn)
    vault = Vault(tmp, Vault.generate_key())
    seed_accounts(sec, {"dr_demo": {"password": "doctor-pass-12345", "role": "doctor", "name": "Dr Demo"},
                        "viewer1": {"password": "viewer-pass-12345", "role": "viewer"}})
    seed_demo_patients(db, vault)
    return db, sec, vault, WebService(db, sec, vault, classifier)


# ------------------------------------------------------------------ security
def test_passwords_and_lockout(tmp):
    h = hash_password("correct horse battery")
    assert h.startswith("scrypt$") and "correct" not in h
    assert verify_password("correct horse battery", h) and not verify_password("wrong", h)
    assert hash_password("x" * 12) != hash_password("x" * 12)          # random salt
    assert not verify_password("a", "not-a-hash")

    db, sec, *_ = make_system(tmp / "a")
    assert sec.authenticate("DR_DEMO ", "doctor-pass-12345").ok            # case/space tolerant
    bad = sec.authenticate("dr_demo", "nope")
    assert not bad.ok and bad.message == "Invalid username or password."
    unknown = sec.authenticate("ghost", "nope")
    assert unknown.message == bad.message                                   # no user enumeration
    for _ in range(MAX_FAILURES):
        sec.authenticate("viewer1", "wrong-password")
    locked = sec.authenticate("viewer1", "viewer-pass-12345")               # even the right one
    assert not locked.ok and "Too many failed attempts" in locked.message
    # expire the lock
    sec.conn.execute("UPDATE login_attempts SET locked_until = ? WHERE username = 'viewer1'",
                     ((datetime.now() - timedelta(minutes=1)).isoformat(),))
    assert sec.authenticate("viewer1", "viewer-pass-12345").ok
    expect(ValueError, sec.upsert_user, "x", "short", "doctor")
    expect(ValueError, sec.upsert_user, "x", "long-enough-pass", "root")
    raw = sec.conn.execute("SELECT password_hash FROM users").fetchall()
    assert all("pass-12345" not in r[0] for r in raw)                       # never stored in clear
    actions = {e["action"] for e in sec.audit_events()}
    assert {"login_ok", "login_failed", "login_blocked"} <= actions


def test_roles(tmp):
    assert can("doctor", "write") and can("doctor", "fhir_send") and can("doctor", "audit")
    assert can("viewer", "read") and can("viewer", "export")
    assert not can("viewer", "write") and not can("viewer", "audit") and not can(None, "read")

    db, sec, vault, svc = make_system(tmp / "b")
    doctor = sec.authenticate("dr_demo", "doctor-pass-12345").user
    viewer = sec.authenticate("viewer1", "viewer-pass-12345").user
    pid = svc.list_patients(viewer)[0]["id"]
    img = sample_scan_image()
    expect(PermissionError, svc.add_visit, viewer, pid, "Nose", img)
    expect(PermissionError, svc.add_treatment, viewer, pid, "Cream", None)
    expect(PermissionError, svc.review, viewer, pid, None, "Agree with the AI result", "No action")
    expect(PermissionError, svc.audit_events, viewer)
    expect(PermissionError, svc.list_patients, None)
    assert svc.open_patient(viewer, pid)["patient"]["id"] == pid           # read is allowed
    assert svc.add_treatment(doctor, pid, "Cream B", None)["treatment_name"] == "Cream B"
    assert len(svc.audit_events(doctor)) > 0


# ------------------------------------------------------------------ vault + uploads
def test_vault_and_upload(tmp):
    vault = Vault(tmp / "v", Vault.generate_key())
    secret = b"patient-image-bytes" * 100
    rel = vault.put(secret)
    assert rel.startswith("vault/") and rel.endswith(".enc")
    assert (tmp / "v" / rel).read_bytes() != secret and secret[:20] not in (tmp / "v" / rel).read_bytes()
    assert vault.get(rel) == secret
    expect(PermissionError, vault.get, "../../etc/passwd")
    other = Vault(tmp / "v", Vault.generate_key())
    expect(ValueError, other.get, rel)                                       # wrong key

    # EXIF / GPS must be removed
    img = synthetic_skin(1, size=(120, 160))
    exif = Image.Exif()
    exif[0x010F] = "SecretPhoneMaker"
    exif[0x8825] = {1: "N", 2: (31.0, 57.0, 0.0)}                            # GPS block
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert b"SecretPhoneMaker" in buf.getvalue()
    clean_img, png = sanitize_image(buf.getvalue())
    assert b"SecretPhoneMaker" not in png and not Image.open(io.BytesIO(png)).getexif()
    assert clean_img.size == (160, 120) or clean_img.size == (120, 160)

    expect(ValueError, sanitize_image, b"")
    expect(ValueError, sanitize_image, b"not an image at all")
    expect(ValueError, sanitize_image, b"x" * (MAX_UPLOAD_BYTES + 1))
    gif = io.BytesIO()
    Image.new("RGB", (10, 10)).save(gif, "GIF")
    expect(ValueError, sanitize_image, gif.getvalue())                       # format whitelist
    big = Image.new("RGB", (4000, 3000), (200, 150, 130))
    b2 = io.BytesIO()
    big.save(b2, "JPEG", quality=30)
    scaled, _ = sanitize_image(b2.getvalue())
    assert max(scaled.size) == 2000


# ------------------------------------------------------------------ service
def test_seed_visits_compare_and_encryption(tmp):
    db, sec, vault, svc = make_system(tmp / "c")
    again = seed_demo_patients(db, vault)
    assert again is False and len(db.search_patients()) == 2                 # idempotent
    doctor = sec.authenticate("dr_demo", "doctor-pass-12345").user
    pa = next(p for p in svc.list_patients(doctor) if p["full_name"] == "Demo Patient A")
    summary = svc.open_patient(doctor, pa["id"])
    assert summary["visit_count"] == 3 and len(summary["timeline"]) >= 4
    assert any("on: Topical cream" in r["text"] for r in summary["timeline"])

    files = list((tmp / "c" / "vault").glob("*.enc"))
    assert len(files) >= 6 and not list((tmp / "c").rglob("*.png")) and not list((tmp / "c").rglob("*.jpg"))
    assert all(f.read_bytes()[:6] == b"gAAAAA" for f in files)                # Fernet tokens

    v1, v3 = summary["visits"][0], summary["visits"][2]
    result, sheet = svc.compare(doctor, v1, v3)
    assert result.aligned and sheet.ndim == 3 and sheet.shape[2] == 3
    assert result.mean_delta_e > 0 and result.changed_percent >= 0
    assert svc.visit_image(v1, "roi").size[0] > 0

    new = svc.add_visit(doctor, pa["id"], "Nose", sample_scan_image(), roi=(100, 80, 300, 200), notes="  new  ")
    assert new["visit_number"] == 4 and new["notes"] == "new"
    expect(ValueError, svc.add_visit, doctor, pa["id"], "Elbow", sample_scan_image())
    expect(ValueError, svc.add_visit, doctor, pa["id"], "Nose", b"junk")
    assert db.integrity_report()["ok"]


def test_scan_and_clinician_review(tmp):
    pred = SimpleNamespace(model_name="fake.tflite", major_class="Eczema", confidence=0.71,
                           uncertain=False, scores={"Eczema": 0.71, "Acne": 0.29}, processing_time=0.04)
    low = SimpleNamespace(model_name="fake.tflite", major_class="Milia", confidence=0.2,
                          uncertain=True, scores={"Milia": 0.2}, processing_time=0.04)
    db, sec, vault, svc = make_system(tmp / "d", classifier=SimpleNamespace(predict=lambda img: pred))
    doctor = sec.authenticate("dr_demo", "doctor-pass-12345").user
    pid = svc.list_patients(doctor)[0]["id"]

    res = svc.classify(doctor, sample_scan_image())
    assert res["top_class"] == "Eczema" and "NOT a diagnosis" not in res["disclaimer"]   # wording check below
    assert "Not a diagnosis" in res["disclaimer"] and "clinician makes every" in res["disclaimer"]
    assert db.get_patient_history(pid) == []                                  # nothing stored by classify()

    cap = svc.save_scan(doctor, pid, sample_scan_image(), res)
    hist = db.get_patient_history(pid)
    assert len(hist) == 1 and hist[0]["predicted_class"] == "Eczema" and hist[0]["is_uncertain"] is False
    assert (tmp / "d" / hist[0]["image_path"]).suffix == ".enc"

    svc.classifier = SimpleNamespace(predict=lambda img: low)
    unc = svc.classify(doctor, sample_scan_image())
    assert unc["uncertain"] is True

    rid = svc.review(doctor, pid, cap, "Disagree with the AI result", "Refer to a dermatologist", "needs biopsy opinion")
    assert rid > 0 and svc.sec.reviews(pid)[0]["username"] == "dr_demo"
    expect(ValueError, svc.review, doctor, pid, cap, "maybe", "No action")
    svc.classifier = None
    expect(RuntimeError, svc.classify, doctor, sample_scan_image())


# ------------------------------------------------------------------ FHIR
def _refs(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "reference":
                yield v
            else:
                yield from _refs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _refs(v)


def _no_empty(obj):
    if isinstance(obj, dict):
        assert obj, "empty object"
        for v in obj.values():
            _no_empty(v)
    elif isinstance(obj, list):
        assert obj, "FHIR forbids empty arrays"
        for v in obj:
            _no_empty(v)


REQUIRED = {"Patient": [], "Encounter": ["status", "class"], "Media": ["status", "content"],
            "Observation": ["status", "code"], "ClinicalImpression": ["status", "subject"],
            "CarePlan": ["status", "intent", "subject"]}


def test_fhir(tmp):
    pred = SimpleNamespace(model_name="m.tflite", major_class="Eczema", confidence=0.71, uncertain=False,
                           scores={"Eczema": 0.71, "Acne": 0.29}, processing_time=0.04)
    db, sec, vault, svc = make_system(tmp / "e", classifier=SimpleNamespace(predict=lambda i: pred))
    doctor = sec.authenticate("dr_demo", "doctor-pass-12345").user
    pa = next(p for p in svc.list_patients(doctor) if p["full_name"] == "Demo Patient A")
    cap = svc.save_scan(doctor, pa["id"], sample_scan_image(), svc.classify(doctor, sample_scan_image()))
    svc.review(doctor, pa["id"], cap, "Agree with the AI result", "Routine follow-up", "")
    stopped = svc.add_treatment(doctor, pa["id"], "Peel (demo)", None)
    svc.stop_treatment(doctor, stopped["id"])

    bundle = build_bundle(db, sec, pa["id"])
    assert bundle["resourceType"] == "Bundle" and bundle["type"] == "transaction"
    json.loads(bundle_json(bundle))                                          # serialises cleanly
    _no_empty(bundle)

    urls = {e["fullUrl"] for e in bundle["entry"]}
    assert len(urls) == len(bundle["entry"]), "fullUrl must be unique"
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    for t in ("Patient", "Encounter", "Media", "Observation", "ClinicalImpression", "CarePlan"):
        assert t in types, t
    assert types.count("Encounter") == 3 and types.count("Media") == 3
    for e in bundle["entry"]:
        r = e["resource"]
        for field in REQUIRED[r["resourceType"]]:
            assert field in r, (r["resourceType"], field)
        assert e["request"]["method"] == "POST" and e["request"]["url"] == r["resourceType"]
        assert e["request"]["ifNoneExist"].startswith(f"identifier={IDENT_SYSTEM}|")
        assert r["identifier"][0]["system"] == IDENT_SYSTEM
    for ref in _refs(bundle):
        assert ref in urls, f"dangling reference {ref}"

    obs = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation")
    assert obs["status"] == "preliminary" and "not a diagnosis" in obs["code"]["text"].lower()
    assert any("Not a diagnosis" in n["text"] for n in obs["note"])
    assert obs["valueString"] == "Eczema"
    plans = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "CarePlan"]
    assert {p["status"] for p in plans} == {"active", "completed"}
    text = bundle_json(bundle).lower()
    assert "base64" not in text and '"data"' not in text and ".enc" not in text   # no images leave
    assert "diagnosis" not in json.dumps([e["resource"] for e in bundle["entry"]
                                          if e["resource"]["resourceType"] == "CarePlan"]).lower()
    assert build_bundle(db, sec, pa["id"])["entry"][0]["fullUrl"] == bundle["entry"][0]["fullUrl"]  # stable ids

    # sending: guard, https only, success and failure paths (network is faked)
    calls = []
    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append((url, headers)); return SimpleNamespace(status_code=201, text="")
    patient = db.get_patient(pa["id"])
    assert is_synthetic(patient)
    out = send_bundle(bundle, patient, post=fake_post)
    assert out["ok"] and calls[0][0] == "https://hapi.fhir.org/baseR4"
    assert calls[0][1]["Content-Type"] == "application/fhir+json"
    assert out["search_url"].endswith(f"Patient?identifier={IDENT_SYSTEM}|{patient['patient_code']}")
    bad = send_bundle(bundle, patient, post=lambda *a, **k: SimpleNamespace(status_code=500, text="boom"))
    assert not bad["ok"] and bad["detail"] == "boom"
    expect(ValueError, send_bundle, bundle, patient, "http://insecure.example/fhir", post=fake_post)

    real = db.add_patient("Real Looking Person", notes="regular patient")    # NOT synthetic
    assert not is_synthetic(db.get_patient(real["id"]))
    expect(PermissionError, send_bundle, build_bundle(db, sec, real["id"]), db.get_patient(real["id"]), post=fake_post)
    assert len(calls) == 1                                                     # nothing was sent for the real one
    assert SYNTHETIC_MARKER in patient["notes"]


# ------------------------------------------------------------------ thermal simulation
def test_thermal_simulation():
    df = simulate_thermal()
    assert list(df.columns) == ["day", "temperature_c", "own_baseline_c", "deviation_c"]
    assert len(df) == 14 and (df["deviation_c"][:7].abs() < 0.6).all()
    assert len(flag_deviation(df, 1.5)) >= 1 and len(flag_deviation(df, 5.0)) == 0
    assert simulate_thermal(seed=5).equals(df)                                 # deterministic


# ------------------------------------------------------------------ add patient, quality check, PDF report
def test_add_patient_quality_and_pdf(tmp):
    from image_quality import assess_quality
    db, sec, vault, svc = make_system(tmp / "g")
    doctor = sec.authenticate("dr_demo", "doctor-pass-12345").user
    viewer = sec.authenticate("viewer1", "viewer-pass-12345").user

    # registering a patient: doctor only, needs the synthetic confirmation, gets the next ID
    before = len(svc.list_patients(doctor))
    expect(PermissionError, svc.add_patient, viewer, "Nope Person", confirm_synthetic=True)
    expect(ValueError, svc.add_patient, doctor, "No Confirm")                      # confirmation missing
    expect(ValueError, svc.add_patient, doctor, "   ", confirm_synthetic=True)     # name required
    expect(ValueError, svc.add_patient, doctor, "Bad Date", date_of_birth="2999-01-01", confirm_synthetic=True)
    expect(ValueError, svc.add_patient, doctor, "Bad Sex", sex="Robot", confirm_synthetic=True)
    assert len(svc.list_patients(doctor)) == before                                 # failures stored nothing
    p = svc.add_patient(doctor, "Test Volunteer", "1990-05-01", "Female", "0790000000", "eczema follow-up",
                        confirm_synthetic=True)
    assert p["patient_code"] == f"P-{before + 1:04d}" and len(svc.list_patients(doctor)) == before + 1
    assert is_synthetic(db.get_patient(p["id"])) and "eczema follow-up" in p["notes"]
    assert any(e["action"] == "add_patient" for e in svc.audit_events(doctor))
    assert [x["id"] for x in svc.list_patients(viewer, "Test Volunteer")] == [p["id"]]   # searchable

    # image quality: a black image and a normal image
    dark = Image.new("RGB", (400, 400), (5, 5, 5))
    assert not assess_quality(dark)["ok"] and any("dark" in w for w in assess_quality(dark)["warnings"])
    good = svc.image_quality(sample_scan_image())
    assert {"brightness", "contrast", "detail", "warnings", "ok"} <= set(good)
    expect(ValueError, svc.image_quality, b"not an image")

    # PDF report: a patient with data, and a brand-new empty patient
    seeded = svc.list_patients(doctor)[0]
    pdf = svc.patient_report(viewer, seeded["id"])                                  # viewers may export
    assert pdf[:5] == b"%PDF-" and len(pdf) > 3000
    empty_pdf = svc.patient_report(doctor, p["id"])
    assert empty_pdf[:5] == b"%PDF-"
    expect(PermissionError, svc.patient_report, None, seeded["id"])
    assert sum(e["action"] == "export_pdf_report" for e in svc.audit_events(doctor)) == 2


def main():
    tests = [test_passwords_and_lockout, test_roles, test_vault_and_upload,
             test_seed_visits_compare_and_encryption, test_scan_and_clinician_review, test_fhir,
             test_add_patient_quality_and_pdf]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        try:
            for t in tests:
                t(Path(tmp) / t.__name__)
                print("ok  ", t.__name__)
        finally:
            for db in OPEN_DBS:
                try:
                    db.close()
                except Exception:
                    pass
    test_thermal_simulation()
    print("ok   test_thermal_simulation")
    print("All web proof-of-concept tests passed.")


if __name__ == "__main__":
    main()
