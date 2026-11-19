# DermaAssist AI: web proof of concept

A small web version of the clinician-side software, for reviewers who cannot see the Raspberry Pi.
**Synthetic demo data only. Research prototype. Not a medical device. Not a diagnosis.**

## What it shows
| Feedback point | What the web app does |
|---|---|
| Patient registration | Doctors register a patient (name, date of birth, sex, phone, notes). Each record is marked synthetic and gets the next ID (P-0001, P-0002...) |
| Secure, accessible web environment | Login (salted scrypt hashes, lock-out after 5 failures, 15-minute idle sign-out), roles (doctor / viewer), audit log |
| Secure storage of patient images | Every stored image is encrypted at rest (Fernet); uploads are re-encoded so EXIF/GPS data is removed |
| EHR / FHIR | One-click **HL7 FHIR R4** export (Patient, Encounter, Media metadata, Observation, ClinicalImpression, CarePlan) and an optional send to the public HAPI test server (synthetic patients only) |
| AI-assisted vs diagnosis | Red "preliminary, not a diagnosis" banner; "Uncertain" below the confidence threshold; AI result exported as an Observation with status *preliminary* |
| Clinicians in full control | The doctor must record an assessment and a decision (no action / follow-up / refer / treat); the platform never changes treatment or referral by itself |
| Image upload | Page **Add visit / upload image**: photo upload, quality check (brightness, contrast, detail), choice of the tracked area, encrypted storage |
| PDF report | One-click patient PDF report (ported from the Raspberry Pi system) and a CSV of the timeline, on the **Doctor dashboard** |
| Longitudinal tracking | Visit timeline, Before/After with alignment and difference map, treatment timeline |
| Thermal data | **Simulation only**, clearly labelled: no sensor exists. Shown only to illustrate the planned module |

## Files included
1. `models/skin_model.tflite`: the classifier (the same file used on the Raspberry Pi).
2. `MODEL_LICENSE.txt`: the model's MIT licence text, required by the licence. See `NOTICE.md`.
3. `packages.txt`: system fonts for the PDF report (Streamlit Community Cloud installs it automatically).
4. Optional `demo_assets/before.jpg` and `demo_assets/after.jpg`: photos of a **consenting** volunteer (no face, tattoo or other identifier). Without them the demo uses generated illustrations.

## Run locally (10 minutes)
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_web.py                                       # all tests should pass
DERMA_DEV=1 streamlit run app.py                         # Windows PowerShell: $env:DERMA_DEV=1; streamlit run app.py
```
`DERMA_DEV=1` creates a temporary `demo_doctor` account and prints its password in the terminal. Never use `DERMA_DEV` on a public deployment.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and push this folder. **Do not push** `data/` or real secrets (the `.gitignore` already excludes them).
2. On share.streamlit.io choose *New app*, select the repository, branch `main`, main file `app.py`.
3. Open *Advanced settings* and choose **Python 3.12**. (LiteRT publishes Linux wheels for Python 3.10 to 3.14.)
4. Paste your secrets (see `.streamlit/secrets.toml.example`):
   * `IMAGE_KEY`: generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
   * one `[accounts.<name>]` block per account, each with a long unique password and role `doctor` or `viewer`.
5. Deploy. The platform provides HTTPS.
6. Open the link in a **private window**, sign in with each account, and run the checks below.

Free apps can go to sleep when nobody uses them and the disk is temporary: patients, visits and images you add are lost on restart, then only the demo patients are recreated. Open the app a few minutes before a reviewer is expected to use it.

### Alternative: Hugging Face Spaces (Streamlit SDK)
Create a Space with the Streamlit SDK and upload the same files. Add `IMAGE_KEY` and `DERMA_ACCOUNTS_JSON` as secrets, for example:
`{"dr_demo": {"password": "...", "role": "doctor", "name": "Demo Doctor"}, "reviewer": {"password": "...", "role": "viewer"}}`

## Give reviewers
* The link, and **two accounts**: a doctor (full demo) and a viewer (read-only). Put the credentials in the submission, not in the repository.
* This 2-minute path:
  1. Sign in as the doctor. Open **Doctor dashboard** for P-0001: timeline, treatment, first vs latest image.
  2. **Follow-up and comparison**: compare visit 1 and 3 (aligned, difference map). Mention it is a pixel difference.
  3. **AI scan**: press *Use the built-in synthetic sample image*. Show the red notice, then record a **clinician decision**.
  4. **EHR / FHIR export**: download the Bundle; optionally send it to the HAPI test server and open the read-back link.
  5. **Audit log** and **About and security**: show what is implemented and, honestly, what is not.
  6. **Thermal monitoring (simulation)**: say clearly that it is simulated.

## Before you submit: checklist
- [ ] `python test_web.py` passes locally.
- [ ] No secrets, no `data/` folder, no real photos in the GitHub repository.
- [ ] Passwords are long and unique; the demo accounts are not reused anywhere else.
- [ ] The app opens in a private browser window and both accounts work.
- [ ] The viewer account cannot add visits, record decisions or open the audit log.
- [ ] Every image used is synthetic or from a consenting volunteer.
- [ ] The submission says: synthetic data only, AI result is preliminary, thermal module is a simulation, no clinical validation.

## Honest limits
No multi-factor authentication, no independent security audit, no backup or retention policy, no regulatory approval, no clinical validation, and no real temperature sensor. The public FHIR test server is open to everyone: send synthetic data only.
