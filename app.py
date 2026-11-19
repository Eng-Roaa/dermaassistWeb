"""
DermaAssist AI - web proof of concept (Streamlit)

Run locally:   DERMA_DEV=1 streamlit run app.py
Deploy:        see README_DEPLOY.md

Synthetic/demo data only. Research prototype: not a medical device, not a diagnosis.
All logic lives in web_service.py, web_security.py and fhir_export.py; this file is the screen layer.
"""

import io
import json
import os
import secrets as pysecrets
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from demo_seed import (flag_deviation, sample_scan_image, seed_accounts, seed_demo_patients,
                       simulate_thermal)
from fhir_export import (DEFAULT_FHIR_BASE, build_bundle, bundle_json, is_synthetic, send_bundle)
from skin_db import SEX_VALUES, SKIN_REGIONS, SkinDB, age_from_dob
from web_security import (AI_AGREEMENT, CLINICIAN_ACTIONS, SecurityStore, Vault, can,
                          sanitize_image)
from web_service import DISCLAIMER, WebService, load_classifier

APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DERMA_DATA_DIR", APP_DIR / "data"))
MODEL_PATH = APP_DIR / "models" / "skin_model.tflite"
IDLE_LIMIT_SECONDS = 15 * 60
PAGE_PATIENTS = "Patients (register / search)"
PAGE_UPLOAD = "Add visit / upload image"
PAGES = [PAGE_PATIENTS, "Doctor dashboard", PAGE_UPLOAD, "Follow-up and comparison", "AI scan (preliminary)",
         "Treatments", "Thermal monitoring (simulation)", "EHR / FHIR export", "Audit log", "About and security"]


# ============================================================ configuration and bootstrap
def secret(name, default=None):
    """Read a Streamlit secret, then an environment variable, else the default."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def configured_accounts() -> dict:
    try:
        return {u: dict(v) for u, v in st.secrets["accounts"].items()}
    except Exception:
        pass
    if os.environ.get("DERMA_ACCOUNTS_JSON"):                    # e.g. Hugging Face Spaces secrets
        return json.loads(os.environ["DERMA_ACCOUNTS_JSON"])
    if os.environ.get("DERMA_DEV") == "1":                       # local development only
        password = pysecrets.token_urlsafe(12)
        print(f"[DERMA_DEV] temporary account  demo_doctor / {password}")
        return {"demo_doctor": {"password": password, "role": "doctor", "name": "Demo Doctor"}}
    return {}


def image_key() -> str:
    key = secret("IMAGE_KEY")
    if key:
        return key
    path = DATA_DIR / ".image_key"                               # development convenience only
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(Vault.generate_key())
    return path.read_text().strip()


@st.cache_resource
def bootstrap() -> dict:
    """Runs once per server process: accounts, demo patients, classifier."""
    db = SkinDB(DATA_DIR / "database" / "web_poc.db")
    sec = SecurityStore(db.conn)
    vault = Vault(DATA_DIR, image_key())
    accounts = configured_accounts()
    seed_accounts(sec, accounts)
    seeded = seed_demo_patients(db, vault, APP_DIR / "demo_assets")
    db.close()
    classifier, reason = load_classifier(MODEL_PATH)
    return {"classifier": classifier, "reason": reason, "accounts": len(accounts), "seeded": seeded,
            "ephemeral_key": secret("IMAGE_KEY") is None}


def get_service(boot: dict):
    """A fresh database connection per script run (SQLite connections are not shared across threads)."""
    db = SkinDB(DATA_DIR / "database" / "web_poc.db")
    sec = SecurityStore(db.conn)
    return WebService(db, sec, Vault(DATA_DIR, image_key()), boot["classifier"]), db


# ============================================================ session
def logout(svc=None, reason=None):
    user = st.session_state.get("user")
    if user and svc:
        svc.sec.log(user["username"], user["role"], "logout", detail=reason or "")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if reason:
        st.session_state["notice"] = reason
    st.rerun()


def enforce_timeout(svc):
    now = time.time()
    last = st.session_state.get("last_active")
    if last and now - last > IDLE_LIMIT_SECONDS:
        logout(svc, "You were signed out after 15 minutes of inactivity.")
    st.session_state["last_active"] = now


def login_screen(svc, boot):
    st.title("DermaAssist AI")
    st.subheader("Clinician portal: skin follow-up proof of concept")
    st.info("Research prototype with **synthetic demo data only**. Not a medical device and not a diagnosis. "
            "Do not enter or upload real patient information.")
    if st.session_state.get("notice"):
        st.warning(st.session_state.pop("notice"))
    if boot["accounts"] == 0:
        st.error("No accounts are configured on this server. The administrator must add accounts to the "
                 "app secrets (see README_DEPLOY.md).")
        return
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        result = svc.sec.authenticate(username, password)
        if result.ok:
            st.session_state["user"] = result.user
            st.session_state["last_active"] = time.time()
            st.rerun()
        else:
            st.error(result.message)


# ============================================================ small helpers
def decrypted(svc, visit, kind="roi"):
    try:
        return svc.visit_image(visit, kind)
    except Exception:
        return None


def banner():
    st.caption("Research prototype · synthetic demo data · AI output is preliminary and never a diagnosis · "
               "the clinician decides")


# ============================================================ pages
def page_dashboard(svc, user, patient_id):
    s = svc.open_patient(user, patient_id)
    p = s["patient"]
    age = age_from_dob(p["date_of_birth"])
    st.header(f"{p['patient_code']} - {p['full_name']}")
    st.caption(f"Age: {age if age is not None else '-'}  |  Sex: {p['sex'] or '-'}  |  {p['notes'] or ''}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI scans", s["scan_count"])
    c2.metric("Tracking visits", s["visit_count"])
    c3.metric("Active treatments", len(s["active_treatments"]))
    c4.metric("Clinician reviews", len(s["reviews"]))

    left, right = st.columns(2)
    with left:
        st.subheader("AI classification summary")
        st.warning(DISCLAIMER)
        if s["analysed_count"]:
            st.write({k: v for k, v in s["class_counts"].items()} or "No confident results yet")
            st.caption(f"Uncertain (low confidence): {s['uncertain_count']}")
        else:
            st.write("No AI scans stored for this patient.")
    with right:
        st.subheader("Follow-up images")
        if s["visit_count"]:
            first, last = s["first_visit"], s["last_visit"]
            a, b = st.columns(2)
            img1, img2 = decrypted(svc, first), decrypted(svc, last)
            if img1:
                a.image(img1, caption=f"First: visit #{first['visit_number']} ({first['visit_date'][:10]})",
                        width="stretch")
            if img2 and s["visit_count"] > 1:
                b.image(img2, caption=f"Latest: visit #{last['visit_number']} ({last['visit_date'][:10]})",
                        width="stretch")
        else:
            st.write("No tracking visits yet.")

    st.subheader("Timeline: visits and treatment")
    timeline_df = pd.DataFrame(s["timeline"])[["date", "kind", "text"]] if s["timeline"] else None
    if timeline_df is not None:
        st.dataframe(timeline_df, hide_index=True, width="stretch")

    st.subheader("Reports and downloads")
    if can(user["role"], "export"):
        pdf_key = f"pdf_{patient_id}_{s['visit_count']}_{s['treatment_count']}_{s['scan_count']}"
        d1, d2 = st.columns(2)
        with d1:
            if st.button("Prepare PDF report"):
                try:
                    with st.spinner("Building the report..."):
                        st.session_state[pdf_key] = svc.patient_report(user, patient_id)
                except Exception as exc:
                    st.error(f"Could not build the report: {exc}")
            if st.session_state.get(pdf_key):
                st.download_button("Download PDF report", st.session_state[pdf_key],
                                   file_name=f"report_{p['patient_code']}.pdf", mime="application/pdf")
        with d2:
            if timeline_df is not None:
                st.download_button("Download timeline (CSV)", timeline_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"timeline_{p['patient_code']}.csv", mime="text/csv")
        st.caption("The FHIR bundle is on the \"EHR / FHIR export\" page.")
    else:
        st.info("Your role cannot export reports.")
    st.subheader("Clinician decisions")
    if s["reviews"]:
        st.dataframe(pd.DataFrame(s["reviews"])[["created_at", "username", "ai_agreement", "action", "note"]],
                     hide_index=True, width="stretch")
    else:
        st.write("No clinician review recorded yet.")


def quality_notice(svc, data: bytes):
    """Technical image-quality hints (brightness, contrast, detail). Not a medical judgement."""
    try:
        q = svc.image_quality(data)
    except ValueError:
        return
    if q["ok"]:
        st.caption(f"Image quality check: no obvious problem (brightness {q['brightness']}/255, "
                   f"contrast {q['contrast']}, detail {q['detail']}). Technical indicators only.")
    else:
        for w in q["warnings"]:
            st.warning(w)
        st.caption("Consider retaking the photo with even lighting and sharp focus. "
                   "These are technical indicators only, not a medical judgement.")


def visit_upload_form(svc, user, patient_id, key):
    """Upload a photo, choose the tracked area, save it as a new encrypted visit."""
    if not can(user["role"], "write"):
        st.info("Your role is read-only. Sign in with a doctor account to upload images.")
        return
    upload = st.file_uploader("Photo (JPEG or PNG, up to 8 MB)", type=["jpg", "jpeg", "png"], key=f"{key}_up")
    region = st.selectbox("Skin region", SKIN_REGIONS, key=f"{key}_region")
    notes = st.text_area("Notes (optional)", key=f"{key}_notes")
    ack = st.checkbox("This image is synthetic or from a consenting volunteer. It is NOT a real, "
                      "identifiable patient.", key=f"{key}_ack")
    if upload is None:
        return
    try:
        img, _ = sanitize_image(upload.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return
    quality_notice(svc, upload.getvalue())
    whole = st.checkbox("Use the whole image", value=False, key=f"{key}_whole")
    roi = None
    if not whole:
        x0, x1 = st.slider("Area: left to right (%)", 0, 100, (25, 75), key=f"{key}_x")
        y0, y1 = st.slider("Area: top to bottom (%)", 0, 100, (25, 75), key=f"{key}_y")
        w, h = img.size
        roi = (int(w * x0 / 100), int(h * y0 / 100), int(w * (x1 - x0) / 100), int(h * (y1 - y0) / 100))
    preview = img.copy()
    preview.thumbnail((700, 700))
    if roi:
        k = preview.size[0] / img.size[0]
        ImageDraw.Draw(preview).rectangle([roi[0] * k, roi[1] * k, (roi[0] + roi[2]) * k, (roi[1] + roi[3]) * k],
                                          outline=(0, 255, 100), width=3)
    st.image(preview, caption="Area to track (green)")
    if st.button("Save visit (encrypted)", disabled=not ack, key=f"{key}_save"):
        try:
            v = svc.add_visit(user, patient_id, region, upload.getvalue(), roi, notes)
            st.success(f"Visit #{v['visit_number']} saved. The image is stored encrypted.")
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))


def page_patients(svc, user, patient_id):
    st.header("Patients")
    tab_list, tab_new = st.tabs(["Registered patients", "Register a new patient"])
    with tab_list:
        query = st.text_input("Search by name or patient ID")
        rows = svc.list_patients(user, query)
        if rows:
            st.dataframe(pd.DataFrame([{
                "Patient ID": p["patient_code"], "Name": p["full_name"],
                "Age": age_from_dob(p["date_of_birth"]), "Sex": p["sex"] or "-",
                "Registered": (p["created_at"] or "")[:10]} for p in rows]),
                hide_index=True, width="stretch")
            st.caption(f"{len(rows)} patient(s). Choose a patient in the sidebar to open the record.")
        else:
            st.info("No patients found." if query else "No patients registered yet.")
    with tab_new:
        if not can(user["role"], "write"):
            st.info("Your role is read-only. Only a clinician (doctor role) can register patients.")
            return
        st.warning("Demo environment: enter synthetic or consenting-volunteer data only. Do not enter real "
                   "patient information. This free hosting has temporary storage, and records can be "
                   "lost when the app restarts.")
        with st.form("new_patient"):
            name = st.text_input("Full name *")
            dob = st.date_input("Date of birth (optional)", value=None, min_value=date(1900, 1, 1),
                                max_value=date.today())
            sex = st.selectbox("Sex (optional)", ["", *SEX_VALUES], format_func=lambda v: v or "-")
            phone = st.text_input("Phone (optional)")
            notes = st.text_area("Notes (optional)")
            confirm = st.checkbox("This is a synthetic or consenting-volunteer record. It is NOT a real, "
                                  "identifiable patient.")
            submitted = st.form_submit_button("Register patient")
        if submitted:
            try:
                p = svc.add_patient(user, name, dob.isoformat() if dob else None, sex or None, phone,
                                    notes, confirm_synthetic=confirm)
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
            else:
                st.session_state["_open_patient"] = p["id"]
                st.session_state["_flash"] = (f"Patient {p['patient_code']} - {p['full_name']} registered. "
                                              "You can now add a visit or an AI scan.")
                st.rerun()


def page_upload(svc, user, patient_id):
    st.header("Add a visit / upload an image")
    p = svc.db.get_patient(patient_id)
    st.caption(f"Patient: {p['patient_code']} - {p['full_name']}. The photo is cleaned (location and device "
               "data removed) and stored encrypted.")
    visit_upload_form(svc, user, patient_id, "up")


def page_followup(svc, user, patient_id):
    st.header("Follow-up and comparison")
    s = svc.open_patient(user, patient_id)
    tab_compare, tab_add = st.tabs(["Compare two visits", "Add a visit (clinician)"])
    with tab_compare:
        visits = s["visits"]
        if len(visits) < 2:
            st.info("At least two visits are needed for a comparison.")
        else:
            labels = {f"Visit #{v['visit_number']} - {v['visit_date'][:10]} - {v['skin_region']}": v for v in visits}
            chosen = st.multiselect("Select exactly two visits", list(labels), default=list(labels)[:1] + list(labels)[-1:])
            if len(chosen) == 2 and st.button("Compare"):
                a, b = sorted((labels[c] for c in chosen), key=lambda v: v["visit_number"])
                if a["skin_region"] != b["skin_region"]:
                    st.warning("These visits are for different regions; the comparison is not meaningful.")
                result, sheet = svc.compare(user, a, b)
                st.image(sheet, width="stretch")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean colour difference", f"{result.mean_delta_e:.1f}")
                m2.metric("Clearly different pixels", f"{result.changed_percent:.1f}%")
                m3.metric("Alignment", "applied" if result.aligned else "not applied")
                for w in result.warnings:
                    st.warning(w)
                st.caption("Pixel difference only: not a clinical measure of improvement or worsening. "
                           "Lighting, angle and distance affect it.")
    with tab_add:
        visit_upload_form(svc, user, patient_id, "fu")


def page_scan(svc, user, patient_id):
    st.header("AI scan (preliminary)")
    st.error("AI-assisted PRELIMINARY classification. NOT a diagnosis. The qualified clinician makes every "
             "diagnostic, treatment and referral decision.")
    if svc.classifier is None:
        st.warning("The classifier is not available on this server. All other functions still work.")
        return
    ack = st.checkbox("This image is synthetic or from a consenting volunteer. It is NOT a real, identifiable patient.")
    upload = st.file_uploader("Skin photo (JPEG or PNG, up to 8 MB)", type=["jpg", "jpeg", "png"], key="scan_up")
    use_sample = st.button("Use the built-in synthetic sample image")
    data = None
    if use_sample:
        data = sample_scan_image()
    elif upload is not None and ack:
        data = upload.getvalue()
    elif upload is not None:
        st.info("Please confirm the image statement above.")
    if data is not None:
        try:
            st.session_state["scan"] = {"bytes": data, "result": svc.classify(user, data)}
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    scan = st.session_state.get("scan")
    if not scan:
        return
    r = scan["result"]
    st.image(Image.open(io.BytesIO(scan["bytes"])), width=360, caption="Analysed image")
    quality_notice(svc, scan["bytes"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Preliminary result", "Uncertain" if r["uncertain"] else r["top_class"])
    c2.metric("Top score", f"{r['confidence'] * 100:.1f}%")
    c3.metric("Analysis time", f"{r['processing_time'] * 1000:.0f} ms")
    if r["uncertain"]:
        st.warning("Uncertain: the top score is below the confidence threshold. Do not rely on this result.")
    st.bar_chart(pd.Series(r["scores"], name="score"))
    st.caption("The model has six classes and no 'healthy skin' class, so it can return a confident label "
               "for an image without a lesion. Model: third-party, MIT licence.")

    st.subheader("Clinician review (required before any use of this result)")
    if not can(user["role"], "write"):
        st.info("Your role is read-only. Only a clinician (doctor role) can record a review.")
        return
    with st.form("review"):
        agree = st.radio("Your assessment of the AI result", AI_AGREEMENT)
        action = st.selectbox("Your decision", CLINICIAN_ACTIONS)
        note = st.text_area("Clinical note (optional)")
        save_image = st.checkbox("Also store the image (encrypted) in this patient's record", value=False)
        done = st.form_submit_button("Record clinician decision")
    if done:
        try:
            capture_id = svc.save_scan(user, patient_id, scan["bytes"], r) if save_image else None
            svc.review(user, patient_id, capture_id, agree, action, note)
            st.success("Decision recorded under your name. Nothing was changed automatically.")
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))


def page_treatments(svc, user, patient_id):
    st.header("Treatments")
    s = svc.open_patient(user, patient_id)
    if s["treatments"]:
        st.dataframe(pd.DataFrame(s["treatments"])[["treatment_name", "start_date", "end_date", "notes"]],
                     hide_index=True, width="stretch")
    else:
        st.write("No treatments recorded.")
    if not can(user["role"], "write"):
        st.info("Your role is read-only.")
        return
    with st.form("add_treatment"):
        name = st.text_input("Treatment or procedure")
        start = st.date_input("Start date", value=date.today(), max_value=date.today())
        visit_labels = {"None": None, **{f"Visit #{v['visit_number']} - {v['visit_date'][:10]}": v["id"] for v in s["visits"]}}
        linked = st.selectbox("Linked visit (optional)", list(visit_labels))
        notes = st.text_area("Notes")
        if st.form_submit_button("Add treatment"):
            try:
                svc.add_treatment(user, patient_id, name, start.isoformat(), notes, visit_labels[linked])
                st.success("Treatment added.")
                st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
    ongoing = {t["treatment_name"] + f" (since {t['start_date']})": t["id"] for t in s["active_treatments"]}
    if ongoing:
        pick = st.selectbox("Stop a treatment", list(ongoing))
        if st.button("Mark as stopped today"):
            svc.stop_treatment(user, ongoing[pick])
            st.rerun()


def page_thermal(svc, user, patient_id):
    st.header("Thermal monitoring (simulation)")
    st.error("SIMULATED DATA. No temperature sensor is connected to this platform. This page only illustrates "
             "how a future thermal module would show a patient's own baseline and a deviation from it. "
             "It is not a detector and has no clinical validity.")
    df = simulate_thermal()
    threshold = st.slider("Alert threshold above the patient's own baseline (degrees C, placeholder value)",
                          0.5, 3.0, 1.5, 0.1)
    st.line_chart(df.set_index("day")[["temperature_c", "own_baseline_c"]])
    flagged = flag_deviation(df, threshold)
    if len(flagged):
        st.warning(f"Illustrative alert (simulation): the simulated reading exceeds the baseline by more than "
                   f"{threshold:.1f} degrees C from day {int(flagged['day'].iloc[0])}. In a real system this would "
                   f"prompt a clinician to examine the area. Thresholds would need clinical validation.")
        st.dataframe(flagged, hide_index=True, width="stretch")
    else:
        st.success("No simulated deviation above the threshold.")
    st.caption("Planned, not built: a real sensor, calibration, validated thresholds and clinical study.")


def page_fhir(svc, user, patient_id):
    st.header("EHR / FHIR export")
    st.write("Exports this patient as an **HL7 FHIR R4 transaction Bundle**: Patient, Encounter, Media (metadata "
             "only), Observation (AI result, status *preliminary*), ClinicalImpression (clinician decisions) and "
             "CarePlan (treatments). **Images are never exported.**")
    patient = svc.db.get_patient(patient_id)
    bundle = build_bundle(svc.db, svc.sec, patient_id)
    counts = pd.Series([e["resource"]["resourceType"] for e in bundle["entry"]]).value_counts()
    st.dataframe(counts.rename("resources"), width="stretch")
    text = bundle_json(bundle)
    with st.expander("Preview the JSON"):
        st.code(text[:6000] + ("\n..." if len(text) > 6000 else ""), language="json")
    if st.download_button("Download FHIR Bundle (JSON)", text, file_name=f"{patient['patient_code']}_fhir_bundle.json",
                          mime="application/fhir+json"):
        svc.sec.log(user["username"], user["role"], "fhir_download", patient["patient_code"])

    st.subheader("Send to a FHIR server (test)")
    if not can(user["role"], "fhir_send"):
        st.info("Only the doctor role can send data to a FHIR server.")
        return
    if not is_synthetic(patient):
        st.error("This patient is not marked as a synthetic demo patient, so sending is blocked.")
        return
    base = st.text_input("FHIR server base URL (HTTPS)", DEFAULT_FHIR_BASE)
    ok = st.checkbox("I understand that the public HAPI test server is open to everyone and may be wiped at any time: "
                     "synthetic data only.")
    if st.button("Send Bundle", disabled=not ok):
        try:
            out = send_bundle(bundle, patient, base)
            svc.sec.log(user["username"], user["role"], "fhir_send", patient["patient_code"], f"status {out['status']}")
            if out["ok"]:
                st.success(f"Server accepted the bundle (HTTP {out['status']}).")
                st.write("Read it back:", out["search_url"])
            else:
                st.error(f"Server answered HTTP {out['status']}: {out['detail']}")
        except Exception as exc:
            st.error(f"Could not send: {exc}")


def page_audit(svc, user, patient_id):
    st.header("Audit log")
    if not can(user["role"], "audit"):
        st.info("Only the doctor role can read the audit log.")
        return
    st.dataframe(pd.DataFrame(svc.audit_events(user, 300)), hide_index=True, width="stretch")


def page_about(svc, user, patient_id):
    st.header("About and security")
    st.subheader("Controls implemented in this proof of concept")
    st.dataframe(pd.DataFrame([
        ("Authentication", "Username and password; passwords stored only as salted scrypt hashes"),
        ("Brute-force protection", "Temporary lock-out after 5 failed attempts; same message for unknown users"),
        ("Roles", "Doctor (read/write/audit/FHIR send) and Viewer (read-only)"),
        ("Session", "Automatic sign-out after 15 minutes of inactivity"),
        ("Images at rest", "Every stored image is encrypted (Fernet: AES-128-CBC + HMAC-SHA256); key kept in secrets"),
        ("Uploads", "JPEG/PNG only, 8 MB limit, metadata (EXIF/GPS) removed, size capped"),
        ("Audit trail", "Logins, views, comparisons, scans, reviews, exports and sends are recorded"),
        ("Transport", "HTTPS provided by the hosting platform"),
        ("Data used", "Synthetic demo data only; real patient data is blocked from the FHIR send function"),
    ], columns=["Control", "Detail"]), hide_index=True, width="stretch")
    st.subheader("Not implemented (be aware)")
    st.write("Multi-factor authentication, independent security audit, backups and retention policy, "
             "regulatory approval, hospital-grade key management, and any clinical validation. "
             "This is a demonstration, not a production medical system.")
    st.subheader("AI and the clinician")
    st.write("The classifier gives a preliminary, experimental result and never a diagnosis. A clinician must "
             "record a decision for any use of it, and the platform never changes treatment or referral by itself.")
    st.subheader("Third-party model")
    st.write("Classifier weights: skin-condition-classifier (EfficientNetV2B0) by Tanishq Shinde, MIT licence. "
             "See NOTICE.md. Its training data is undisclosed; it is not cleared for clinical use.")


PAGE_FUNCS = {PAGE_PATIENTS: page_patients, "Doctor dashboard": page_dashboard, PAGE_UPLOAD: page_upload,
              "Follow-up and comparison": page_followup, "AI scan (preliminary)": page_scan,
              "Treatments": page_treatments, "Thermal monitoring (simulation)": page_thermal,
              "EHR / FHIR export": page_fhir, "Audit log": page_audit, "About and security": page_about}


# ============================================================ main
def main():
    st.set_page_config(page_title="DermaAssist AI - clinician portal", page_icon="🩺", layout="wide")
    boot = bootstrap()
    svc, db = get_service(boot)
    try:
        user = st.session_state.get("user")
        if not user:
            login_screen(svc, boot)
            return
        enforce_timeout(svc)
        with st.sidebar:
            st.markdown(f"**{user['display_name']}**  \nRole: `{user['role']}`")
            if st.button("Sign out"):
                logout(svc)
            st.divider()
            patients = svc.list_patients(user)
            labels = {f"{p['patient_code']} - {p['full_name']}": p["id"] for p in patients}
            pending = st.session_state.pop("_open_patient", None)
            for label, pid in labels.items():
                if pending is not None and pid == pending:
                    st.session_state["patient_pick"] = label
            chosen = st.selectbox("Patient", list(labels), key="patient_pick") if labels else None
            pages = PAGES if labels else [PAGE_PATIENTS]
            page = st.radio("Navigate", pages, key="page_pick")
            if boot["ephemeral_key"]:
                st.caption("Image key generated locally (development). Set IMAGE_KEY for deployment.")
        banner()
        flash = st.session_state.pop("_flash", None)
        if flash:
            st.success(flash)
        if not labels:
            st.info("No patients yet. Register the first patient below." if can(user["role"], "write")
                    else "No patients yet.")
        PAGE_FUNCS[page](svc, user, labels[chosen] if chosen else None)
    finally:
        db.close()


main()
