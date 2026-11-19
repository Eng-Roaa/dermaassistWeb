"""
skin_db.py  -  SQLite storage for the Skin Assessment project (Day 1)

Tables
------
patients     one row per patient, with a human-readable Patient ID (P-0001)
captures     one row per photo taken for a patient (path stored relative to BASE)
predictions  one row per model run on a capture (a capture may be re-analysed
             later with a newer model, so this is 1-to-many)

Usage
-----
    db = SkinDB()
    patient = db.add_patient("Ahmad Ali", date_of_birth="1998-04-12", sex="Male")
    print(patient["patient_code"])                      # P-0001
    cap_id = db.add_capture(patient["id"], "images/P-0001/p.jpg")
    db.add_prediction(cap_id, "skin_model.tflite", "Eczema", 0.78, False,
                      {"Eczema": 0.78, "Acne": 0.1}, 0.04)
    history = db.get_patient_history(patient["id"])
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import date, datetime
from pathlib import Path

BASE = Path.home() / "skin_ai_project"
DB_PATH = BASE / "database" / "skin_assessment.db"

SCHEMA_VERSION = 3

SEX_VALUES = ("Male", "Female", "Other")

SKIN_REGIONS = (
    "Forehead", "Left Cheek", "Right Cheek", "Nose",
    "Left Arm", "Right Arm", "Other",
)

SCHEMA_V1 = """
CREATE TABLE patients (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code  TEXT UNIQUE,
    full_name     TEXT NOT NULL CHECK (length(trim(full_name)) > 0),
    date_of_birth TEXT,
    sex           TEXT CHECK (sex IS NULL OR sex IN ('Male','Female','Other')),
    phone         TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE captures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL
                REFERENCES patients(id) ON DELETE RESTRICT,
    image_path  TEXT NOT NULL,          -- relative to BASE, e.g. images/P-0001/x.jpg
    body_site   TEXT,
    notes       TEXT,
    captured_at TEXT NOT NULL
);

CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id      INTEGER NOT NULL
                    REFERENCES captures(id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    is_uncertain    INTEGER NOT NULL DEFAULT 0 CHECK (is_uncertain IN (0,1)),
    scores_json     TEXT NOT NULL,      -- {"Acne": 0.01, "Eczema": 0.78, ...}
    processing_time REAL,
    created_at      TEXT NOT NULL
);

CREATE INDEX idx_captures_patient   ON captures(patient_id, captured_at);
CREATE INDEX idx_predictions_capture ON predictions(capture_id);
CREATE INDEX idx_patients_name       ON patients(full_name);
"""

# Day 2: Skin Tracker. Visits are append-only (the code has no update/delete).
SCHEMA_V2 = """
CREATE TABLE visits (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL
                        REFERENCES patients(id) ON DELETE RESTRICT,
    visit_number        INTEGER NOT NULL,          -- 1, 2, 3 ... per patient
    visit_date          TEXT NOT NULL,
    skin_region         TEXT NOT NULL,
    original_image_path TEXT NOT NULL,             -- relative to BASE, never modified
    image_width         INTEGER NOT NULL,
    image_height        INTEGER NOT NULL,
    roi_image_path      TEXT,                      -- cropped region (optional)
    roi_x               INTEGER,
    roi_y               INTEGER,
    roi_w               INTEGER,
    roi_h               INTEGER,
    notes               TEXT,
    UNIQUE (patient_id, visit_number),
    CHECK (
        (roi_image_path IS NULL AND roi_x IS NULL AND roi_y IS NULL
                                AND roi_w IS NULL AND roi_h IS NULL)
        OR
        (roi_image_path IS NOT NULL AND roi_x >= 0 AND roi_y >= 0
                                    AND roi_w > 0 AND roi_h > 0)
    )
);

CREATE INDEX idx_visits_patient ON visits(patient_id, visit_number);
"""

# Day 3: Treatment history. Like visits, treatment rows are never deleted; the
# only change allowed afterwards is setting the end date once (stop_treatment).
SCHEMA_V3 = """
CREATE TABLE treatments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL
                   REFERENCES patients(id) ON DELETE RESTRICT,
    visit_id       INTEGER
                   REFERENCES visits(id) ON DELETE RESTRICT,   -- optional link
    treatment_name TEXT NOT NULL CHECK (length(trim(treatment_name)) > 0),
    start_date     TEXT NOT NULL,                              -- YYYY-MM-DD
    end_date       TEXT,                                       -- NULL = ongoing
    notes          TEXT,
    created_at     TEXT NOT NULL,
    CHECK (end_date IS NULL OR end_date >= start_date)
);

CREATE INDEX idx_treatments_patient ON treatments(patient_id, start_date);
"""

MIGRATIONS = {1: SCHEMA_V1, 2: SCHEMA_V2, 3: SCHEMA_V3}


def now_iso() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def age_from_dob(dob: str | None) -> int | None:
    """Return age in years from a YYYY-MM-DD string, or None if unknown."""
    if not dob:
        return None
    try:
        born = datetime.strptime(dob, "%Y-%m-%d").date()
    except ValueError:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _clean(value: str | None) -> str | None:
    """Strip whitespace; turn empty strings into NULL."""
    if value is None:
        return None
    value = value.strip()
    return value or None


class SkinDB:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # SQLite does not enforce foreign keys unless asked, per connection.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # ------------------------------------------------------------------ setup
    def _init_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema v{version} is newer than this code "
                f"(v{SCHEMA_VERSION}). Update the application."
            )
        if 0 < version < SCHEMA_VERSION:
            # Existing data: keep a copy before changing the schema.
            backup = self.db_path.with_name(
                f"{self.db_path.stem}.backup-v{version}{self.db_path.suffix}")
            if not backup.exists():
                shutil.copy2(self.db_path, backup)
        for target in range(version + 1, SCHEMA_VERSION + 1):
            self.conn.executescript(MIGRATIONS[target])
            self.conn.execute(f"PRAGMA user_version = {target}")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --------------------------------------------------------------- patients
    def add_patient(
        self,
        full_name: str,
        date_of_birth: str | None = None,
        sex: str | None = None,
        phone: str | None = None,
        notes: str | None = None,
    ) -> dict:
        """Create a patient and assign the next Patient ID (P-0001, P-0002, ...)."""
        full_name = _clean(full_name)
        if not full_name:
            raise ValueError("Patient name is required.")

        date_of_birth = _clean(date_of_birth)
        if date_of_birth:
            try:
                parsed = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError("Date of birth must be in YYYY-MM-DD format.")
            if parsed > date.today():
                raise ValueError("Date of birth cannot be in the future.")

        sex = _clean(sex)
        if sex and sex not in SEX_VALUES:
            raise ValueError(f"Sex must be one of: {', '.join(SEX_VALUES)}.")

        # One transaction: insert, then derive the code from the new row id.
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO patients "
                "(full_name, date_of_birth, sex, phone, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (full_name, date_of_birth, sex, _clean(phone), _clean(notes), now_iso()),
            )
            patient_id = cur.lastrowid
            code = f"P-{patient_id:04d}"
            self.conn.execute(
                "UPDATE patients SET patient_code = ? WHERE id = ?", (code, patient_id)
            )
        return self.get_patient(patient_id)

    def get_patient(self, patient_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_patient_by_code(self, code: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM patients WHERE patient_code = ?", (code.strip().upper(),)
        ).fetchone()
        return dict(row) if row else None

    def search_patients(self, query: str = "") -> list[dict]:
        """List patients (newest first) with scan count and last scan time.
        `query` matches the name, Patient ID or phone number."""
        sql = """
            SELECT p.*,
                   COUNT(c.id)        AS scan_count,
                   MAX(c.captured_at) AS last_scan
            FROM patients p
            LEFT JOIN captures c ON c.patient_id = p.id
        """
        params: tuple = ()
        query = (query or "").strip()
        if query:
            esc = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like = f"%{esc}%"
            sql += (
                " WHERE p.full_name LIKE ? ESCAPE '\\'"
                " OR p.patient_code LIKE ? ESCAPE '\\'"
                " OR p.phone LIKE ? ESCAPE '\\'"
            )
            params = (like, like, like)
        sql += " GROUP BY p.id ORDER BY p.id DESC"
        return [dict(r) for r in self.conn.execute(sql, params)]

    # --------------------------------------------------------------- captures
    def add_capture(
        self,
        patient_id: int,
        image_path: str,
        body_site: str | None = None,
        notes: str | None = None,
        captured_at: str | None = None,
    ) -> int:
        """Link a saved image to a patient. `image_path` should be relative to BASE."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO captures "
                "(patient_id, image_path, body_site, notes, captured_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (patient_id, str(image_path), _clean(body_site), _clean(notes),
                 captured_at or now_iso()),
            )
        return cur.lastrowid

    # ------------------------------------------------------------ predictions
    def add_prediction(
        self,
        capture_id: int,
        model_name: str,
        predicted_class: str,
        confidence: float,
        is_uncertain: bool,
        scores: dict[str, float],
        processing_time: float | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO predictions "
                "(capture_id, model_name, predicted_class, confidence, is_uncertain, "
                " scores_json, processing_time, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (capture_id, model_name, predicted_class, float(confidence),
                 int(bool(is_uncertain)), json.dumps(scores), processing_time,
                 now_iso()),
            )
        return cur.lastrowid

    # ---------------------------------------------------------------- history
    def get_patient_history(self, patient_id: int) -> list[dict]:
        """All captures for a patient, newest first, each with its latest prediction
        (prediction fields are None if the model never ran on that capture)."""
        rows = self.conn.execute(
            """
            SELECT c.id AS capture_id, c.image_path, c.body_site, c.notes,
                   c.captured_at,
                   p.id AS prediction_id, p.model_name, p.predicted_class,
                   p.confidence, p.is_uncertain, p.scores_json, p.processing_time
            FROM captures c
            LEFT JOIN predictions p
                   ON p.id = (SELECT MAX(id) FROM predictions
                              WHERE capture_id = c.id)
            WHERE c.patient_id = ?
            ORDER BY c.captured_at DESC, c.id DESC
            """,
            (patient_id,),
        ).fetchall()

        history = []
        for r in rows:
            item = dict(r)
            raw = item.pop("scores_json")
            item["scores"] = json.loads(raw) if raw else None
            item["is_uncertain"] = (
                bool(item["is_uncertain"]) if item["is_uncertain"] is not None else None
            )
            history.append(item)
        return history

    # ----------------------------------------------------------------- visits
    def add_visit(
        self,
        patient_id: int,
        skin_region: str,
        original_image_path: str,
        image_size: tuple[int, int],
        roi: tuple[int, int, int, int] | None = None,
        roi_image_path: str | None = None,
        notes: str | None = None,
        visit_date: str | None = None,
    ) -> dict:
        """Add a tracking visit. The visit number is the patient's next number.
        `roi` is (x, y, w, h) in pixels of the original image."""
        skin_region = _clean(skin_region)
        if skin_region not in SKIN_REGIONS:
            raise ValueError("Please choose a skin region from the list.")
        width, height = image_size
        if roi is not None:
            x, y, w, h = roi
            if not roi_image_path:
                raise ValueError("roi_image_path is required when roi is given.")
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError("ROI must be inside the image.")
            roi_values = (int(x), int(y), int(w), int(h))
        else:
            roi_values = (None, None, None, None)
            roi_image_path = None

        with self.conn:
            number = self.conn.execute(
                "SELECT COALESCE(MAX(visit_number), 0) + 1 FROM visits "
                "WHERE patient_id = ?", (patient_id,)
            ).fetchone()[0]
            cur = self.conn.execute(
                "INSERT INTO visits (patient_id, visit_number, visit_date, "
                " skin_region, original_image_path, image_width, image_height, "
                " roi_image_path, roi_x, roi_y, roi_w, roi_h, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (patient_id, number, visit_date or now_iso(), skin_region,
                 str(original_image_path), int(width), int(height),
                 roi_image_path, *roi_values, _clean(notes)),
            )
        return self.get_visit(cur.lastrowid)

    def get_visit(self, visit_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
        return dict(row) if row else None

    def list_visits(self, patient_id: int) -> list[dict]:
        """All tracking visits of one patient, oldest first (Visit 1, 2, 3 ...)."""
        rows = self.conn.execute(
            "SELECT * FROM visits WHERE patient_id = ? "
            "ORDER BY visit_number", (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- treatments
    @staticmethod
    def _valid_day(value: str | None, label: str) -> str | None:
        value = _clean(value)
        if value is None:
            return None
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"{label} must be in YYYY-MM-DD format.")
        return value

    def add_treatment(
        self,
        patient_id: int,
        treatment_name: str,
        start_date: str | None = None,
        notes: str | None = None,
        visit_id: int | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Record a treatment/procedure for a patient, optionally linked to a visit."""
        name = _clean(treatment_name)
        if not name:
            raise ValueError("Treatment name is required.")
        start = self._valid_day(start_date, "Start date") or date.today().isoformat()
        end = self._valid_day(end_date, "End date")
        if start > date.today().isoformat():
            raise ValueError("Start date cannot be in the future.")
        if end and end < start:
            raise ValueError("End date cannot be before the start date.")
        if visit_id is not None:
            owner = self.conn.execute(
                "SELECT patient_id FROM visits WHERE id = ?", (visit_id,)).fetchone()
            if owner is None or owner["patient_id"] != patient_id:
                raise ValueError("The selected visit does not belong to this patient.")

        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO treatments (patient_id, visit_id, treatment_name, "
                " start_date, end_date, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (patient_id, visit_id, name, start, end, _clean(notes), now_iso()),
            )
        return self.get_treatment(cur.lastrowid)

    def stop_treatment(self, treatment_id: int, end_date: str | None = None) -> dict:
        """Mark an ongoing treatment as finished. Can be done only once."""
        row = self.get_treatment(treatment_id)
        if row is None:
            raise ValueError("Treatment not found.")
        if row["end_date"] is not None:
            raise ValueError("This treatment already has an end date.")
        end = self._valid_day(end_date, "End date") or date.today().isoformat()
        if end < row["start_date"]:
            raise ValueError("End date cannot be before the start date.")
        with self.conn:
            self.conn.execute(
                "UPDATE treatments SET end_date = ? WHERE id = ? AND end_date IS NULL",
                (end, treatment_id))
        return self.get_treatment(treatment_id)

    def get_treatment(self, treatment_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM treatments WHERE id = ?", (treatment_id,)).fetchone()
        return dict(row) if row else None

    def list_treatments(self, patient_id: int) -> list[dict]:
        """A patient's treatments, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM treatments WHERE patient_id = ? "
            "ORDER BY start_date, id", (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- integrity
    def integrity_report(self) -> dict:
        """Database health: SQLite's own check plus foreign-key violations."""
        integrity = self.conn.execute("PRAGMA integrity_check").fetchall()
        violations = self.conn.execute("PRAGMA foreign_key_check").fetchall()
        return {
            "ok": [r[0] for r in integrity] == ["ok"] and not violations,
            "integrity": [r[0] for r in integrity],
            "foreign_key_violations": [tuple(r) for r in violations],
            "schema_version": self.conn.execute("PRAGMA user_version").fetchone()[0],
        }
