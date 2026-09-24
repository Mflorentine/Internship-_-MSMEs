import os
import json
import sqlite3
import datetime as dt
import secrets as secrets_lib
import smtplib
from email.mime.text import MIMEText

import bcrypt
import joblib
import numpy as np
import pyotp
import qrcode
import io
import streamlit as st
from streamlit_option_menu import option_menu

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="Sepsis Risk Assistant",
                    layout="wide",
                    page_icon="🩺")

working_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = f"{working_dir}/sepsis_cases.db"

# -------------------------------------------------------------------
# Loading the saved model (produced in Week 8 of the internship notebook)
# -------------------------------------------------------------------
sepsis_model = joblib.load(f'{working_dir}/final_sepsis_model.pkl')

try:
    with open(f'{working_dir}/model_feature_documentation.json') as f:
        model_doc = json.load(f)
    FEATURES = model_doc["features"]
    THRESHOLD = float(model_doc.get("classification_threshold", 0.5))
except FileNotFoundError:
    FEATURES = ['Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                'BaseExcess', 'FiO2', 'pH', 'Glucose', 'Potassium', 'Hct',
                'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
    THRESHOLD = 0.5

FIELD_LABELS = {
    'Hour':         'Hour of ICU stay recorded',
    'HR':           'Heart Rate (beats/min)',
    'O2Sat':        'Oxygen Saturation (%)',
    'Temp':         'Temperature (°C)',
    'SBP':          'Systolic Blood Pressure (mmHg)',
    'MAP':          'Mean Arterial Pressure (mmHg)',
    'DBP':          'Diastolic Blood Pressure (mmHg)',
    'Resp':         'Respiration Rate (breaths/min)',
    'BaseExcess':   'Base Excess (mmol/L)',
    'FiO2':         'Fraction of Inspired Oxygen',
    'pH':           'Arterial pH',
    'Glucose':      'Glucose (mg/dL)',
    'Potassium':    'Potassium (mmol/L)',
    'Hct':          'Hematocrit (%)',
    'Age':          'Age (years)',
    'Gender':       'Gender (1 = Male, 0 = Female)',
    'Unit1':        'Admitted to MICU (1 = Yes, 0 = No)',
    'Unit2':        'Admitted to SICU (1 = Yes, 0 = No)',
    'HospAdmTime':  'Hours between hospital and ICU admission',
    'ICULOS':       'ICU Length of Stay (hours)',
}

FIELD_GROUPS = [
    ("Vital Signs", "🫀", ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']),
    ("Labs & Blood Gas", "🧪", ['BaseExcess', 'FiO2', 'pH', 'Glucose', 'Potassium', 'Hct']),
    ("Patient & Admission Info", "🧍", ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'Hour', 'ICULOS']),
]

CLINICAL_GUIDANCE = {
    "HIGH RISK": [
        "Escalate for prompt physician / rapid-response review.",
        "Consider blood cultures, lactate, and broad-spectrum antibiotics per your unit's sepsis bundle.",
        "Increase vitals monitoring frequency and reassess trend, not just this single reading.",
    ],
    "MODERATE RISK": [
        "Recommend repeat vitals and reassessment within the next hour.",
        "Review recent trend in vitals/labs, not just the current snapshot.",
        "Flag to the care team for awareness; consider closer observation.",
    ],
    "LOW RISK": [
        "Continue routine monitoring per your unit's standard protocol.",
        "Re-screen if the patient's clinical picture changes.",
    ],
}

# ---------------------------------------------------------------------------
# ACCOUNTS -- two sources, merged at login time:
#  1. SECRET_USERS: bootstrap accounts from Streamlit secrets (immutable from inside
#     the app -- st.secrets cannot be written to at runtime). At minimum, this must
#     contain one Admin account so someone can create everyone else.
#  2. Accounts created or password-reset later via the in-app Admin panel, stored in
#     the user_accounts table. These take precedence over a secrets-defined account
#     with the same username, so an Admin can rotate even a bootstrap password.
# Passwords are never stored or compared in plaintext: every account holds a bcrypt
# hash, checked with bcrypt.checkpw() at login time.
# ---------------------------------------------------------------------------
# DEFAULT_DEMO_USERS: built-in accounts so the app is usable immediately, without
# anyone having to set up a secrets.toml file or Streamlit Cloud Secrets panel first.
# These are for demoing/testing only -- change these passwords (or remove this dict
# entirely and switch to secrets.toml) before using this with any real patient data.
# Same bcrypt-hash-and-checkpw scheme as every other account; nothing is stored in
# plaintext.
#   admin1     / admin123
#   clinician1 / clinician123
#   doctor1    / doctor123
DEFAULT_DEMO_USERS = {
    "admin1": {
        "password_hash": "$2b$12$AOdFx7ABJGne.TLFa.OVI.si5PQxcj3PvrufA7rGVf0Agp1E65Vmy",
        "role": "Admin",
        "name": "Demo Admin",
        "email": "muflorentine4@gmail.com",
    },
    "clinician1": {
        "password_hash": "$2b$12$EikCMR7q3fUWqCFKUyM09uJXrQV.lh7VE52HngSM5x3uwLQti5b8m",
        "role": "Clinician",
        "name": "florentinemukamana@gmail.com",
    },
    "doctor1": {
        "password_hash": "$2b$12$mk0fBVwgcNwNvqcpmrkdH.uG5pbp3gmrlMMPImbeesm2lAYJaVa/6",
        "role": "Doctor",
        "name": "muflorentine3@gmail.com",
    },
}

try:
    _CONFIGURED_SECRET_USERS = {uname: dict(udata) for uname, udata in st.secrets["users"].items()}
except (KeyError, FileNotFoundError, AttributeError):
    _CONFIGURED_SECRET_USERS = {}

# Secrets-configured accounts take precedence over the built-in demo ones if a
# username collides (e.g. you define your own "clinician1" in secrets.toml later).
SECRET_USERS = {**DEFAULT_DEMO_USERS, **_CONFIGURED_SECRET_USERS}


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed or missing hash for this account -- fail closed, never fall back to
        # a plaintext comparison.
        return False


def get_all_users():
    """Merge bootstrap accounts (Streamlit secrets) with accounts created or
    password-reset at runtime via the Admin panel (stored in SQLite). Called fresh
    on every login attempt, not cached, so a newly created account works immediately
    without redeploying the app."""
    merged = dict(SECRET_USERS)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM user_accounts").fetchall()
    except sqlite3.OperationalError:
        rows = []  # table not created yet on a brand-new database
    conn.close()
    for row in rows:
        merged[row["username"]] = {
            "password_hash": row["password_hash"],
            "role": row["role"],
            "name": row["name"],
            "email": row["email"] if "email" in row.keys() else None,
        }
    return merged


def create_or_reset_account(username, password, role, name, email=None):
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO user_accounts (username, password_hash, role, name, email, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(username) DO UPDATE SET password_hash = excluded.password_hash, "
        "role = excluded.role, name = excluded.name, "
        "email = COALESCE(excluded.email, user_accounts.email)",
        (username, password_hash, role, name, email, dt.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def set_account_password(username, new_password):
    """Used by the self-service reset flows (authenticator code or email link) --
    updates only the password, leaving role/name/email untouched. Works whether the
    account originally came from Streamlit secrets or the Admin panel, by copying
    its current role/name/email into user_accounts (which always wins in
    get_all_users)."""
    users = get_all_users()
    account = users.get(username)
    if not account:
        return False
    create_or_reset_account(
        username=username,
        password=new_password,
        role=account["role"],
        name=account.get("name", username),
        email=account.get("email"),
    )
    return True


def fetch_all_accounts_overview():
    """For the Admin panel's account list -- shows every account and where it's
    defined, without ever exposing a password or password hash."""
    accounts = []
    for uname, udata in SECRET_USERS.items():
        accounts.append({"username": uname, "name": udata.get("name", ""), "role": udata.get("role", ""),
                          "email": udata.get("email") or "(not set)", "source": "Streamlit secrets"})
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT username, name, role, email FROM user_accounts").fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    db_usernames = {r["username"] for r in rows}
    accounts = [a for a in accounts if a["username"] not in db_usernames]  # DB overrides secrets
    for row in rows:
        accounts.append({"username": row["username"], "name": row["name"], "role": row["role"],
                          "email": row["email"] or "(not set)", "source": "Created/reset via Admin panel"})
    return accounts


# -------------------------------------------------------------------
# Self-service password reset: authenticator (TOTP) codes and email links
# -------------------------------------------------------------------
RESET_TOKEN_LIFETIME_MINUTES = 30


def get_totp_secret(username):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT secret, enabled FROM user_totp WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def start_totp_enrollment(username):
    """Generates a brand-new secret and stores it as not-yet-enabled. Nothing takes
    effect until confirm_totp_enrollment verifies the person actually scanned it and
    can produce a valid code -- otherwise a half-finished setup could lock them out."""
    secret = pyotp.random_base32()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO user_totp (username, secret, enabled, created_at) VALUES (?, ?, 0, ?) "
        "ON CONFLICT(username) DO UPDATE SET secret = excluded.secret, enabled = 0",
        (username, secret, dt.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return secret


def confirm_totp_enrollment(username, code):
    row = get_totp_secret(username)
    if not row or not pyotp.TOTP(row["secret"]).verify(code, valid_window=1):
        return False
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE user_totp SET enabled = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return True


def verify_totp_code(username, code):
    row = get_totp_secret(username)
    if not row or not row["enabled"]:
        return False
    return pyotp.TOTP(row["secret"]).verify(code, valid_window=1)


def totp_qr_png_bytes(username, secret):
    uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name="Sepsis Risk Assistant")
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_password_reset_token(username):
    token = secrets_lib.token_urlsafe(32)
    now = dt.datetime.now()
    expires = now + dt.timedelta(minutes=RESET_TOKEN_LIFETIME_MINUTES)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO password_resets (token, username, created_at, expires_at, used) "
        "VALUES (?, ?, ?, ?, 0)",
        (token, username, now.isoformat(timespec="seconds"), expires.isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return token


def get_valid_reset_token(token):
    """Returns the username for a token that exists, hasn't been used, and hasn't
    expired -- or None if any of that fails, without distinguishing which (so a
    guessed/expired/reused token all just look like 'invalid link')."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT username, expires_at, used FROM password_resets WHERE token = ?", (token,)
    ).fetchone()
    conn.close()
    if not row or row["used"]:
        return None
    if dt.datetime.now() > dt.datetime.fromisoformat(row["expires_at"]):
        return None
    return row["username"]


def consume_reset_token(token):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE password_resets SET used = 1 WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def send_reset_email(to_email, reset_link):
    """Sends the reset link over SMTP using credentials from st.secrets['smtp'].
    Returns (success, message) rather than raising, so the caller can show a clean
    error in the UI instead of a stack trace when SMTP isn't configured yet."""
    try:
        smtp_cfg = st.secrets["smtp"]
    except (KeyError, FileNotFoundError, AttributeError):
        return False, ("Email sending isn't configured yet. An administrator needs to add "
                        "an [smtp] section to this app's Secrets before reset links can be sent.")
    body = (
        "You (or someone else) requested a password reset for the Sepsis Risk Assistant.\n\n"
        f"Reset your password here (link expires in {RESET_TOKEN_LIFETIME_MINUTES} minutes):\n"
        f"{reset_link}\n\n"
        "If you didn't request this, you can safely ignore this email."
    )
    msg = MIMEText(body)
    msg["Subject"] = "Reset your Sepsis Risk Assistant password"
    msg["From"] = smtp_cfg["sender"]
    msg["To"] = to_email
    try:
        with smtplib.SMTP(smtp_cfg["host"], int(smtp_cfg.get("port", 587))) as server:
            server.starttls()
            server.login(smtp_cfg["username"], smtp_cfg["password"])
            server.sendmail(smtp_cfg["sender"], [to_email], msg.as_string())
        return True, "Reset link sent."
    except Exception as e:
        return False, f"Couldn't send the email: {e}"


# Formal role-permission mapping (RBAC), checked server-side wherever a protected
# action happens, rather than scattering ad hoc "if role == ..." checks throughout
# the page logic. Mirrors the authorization.py pattern in the authentication spec.
ROLE_PERMISSIONS = {
    "Clinician": {"submit_assessment", "view_own_submissions"},
    "Doctor": {"review_assessments", "view_all_submissions", "view_auth_activity"},
    "Admin": {"manage_accounts", "view_auth_activity"},
}


def has_permission(role, permission):
    return permission in ROLE_PERMISSIONS.get(role, set())

# -------------------------------------------------------------------
# Lightweight local database for the clinician -> doctor handoff.
# Note: on Streamlit Community Cloud, local disk storage is ephemeral and is
# wiped on redeploy or after the app sleeps -- fine for a class/internship
# demo, but a real deployment needs a persistent external database.
# -------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_label TEXT,
            submitted_by TEXT,
            submitted_at TEXT,
            probability REAL,
            risk_label TEXT,
            inputs_json TEXT,
            reviewed INTEGER DEFAULT 0,
            doctor_note TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            event_type TEXT,
            event_status TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            role TEXT,
            name TEXT,
            created_at TEXT
        )
    """)
    # Added later than the original table -- ALTER only succeeds once per DB file,
    # so an existing deployment's table gets the column added on next startup
    # instead of needing a fresh database.
    try:
        conn.execute("ALTER TABLE user_accounts ADD COLUMN email TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_totp (
            username TEXT PRIMARY KEY,
            secret TEXT NOT NULL,
            enabled INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at TEXT,
            expires_at TEXT,
            used INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_auth_event(username, event_type, event_status):
    """Record an authentication event (LOGIN/LOGOUT, SUCCESS/FAILED) to a dedicated
    audit table, separate from clinical data -- never logs passwords or password hashes."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO auth_audit (username, event_type, event_status, created_at) VALUES (?, ?, ?, ?)",
        (username, event_type, event_status, dt.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def fetch_auth_audit(limit=25):
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT created_at, username, event_type, event_status FROM auth_audit "
        "ORDER BY created_at DESC LIMIT ?",
        conn, params=(limit,),
    )
    conn.close()
    return df


def save_assessment(patient_label, submitted_by, probability, risk_label, inputs):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO assessments (patient_label, submitted_by, submitted_at, probability, "
        "risk_label, inputs_json) VALUES (?, ?, ?, ?, ?, ?)",
        (patient_label, submitted_by, dt.datetime.now().isoformat(timespec="seconds"),
         probability, risk_label, json.dumps(inputs)),
    )
    conn.commit()
    conn.close()


def fetch_assessments(only_unreviewed=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM assessments"
    if only_unreviewed:
        query += " WHERE reviewed = 0"
    query += " ORDER BY submitted_at DESC"
    rows = conn.execute(query).fetchall()
    conn.close()
    return rows


def mark_reviewed(assessment_id, note=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE assessments SET reviewed = 1, doctor_note = ? WHERE id = ?",
                 (note, assessment_id))
    conn.commit()
    conn.close()


def fetch_all_as_csv_bytes():
    """Export every row in the assessments table as CSV bytes, for backup/reporting.
    Important given Streamlit Cloud's local storage is wiped on redeploy/sleep."""
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM assessments ORDER BY submitted_at DESC", conn)
    conn.close()
    return df.to_csv(index=False).encode("utf-8"), len(df)


def fetch_by_submitter(submitted_by):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM assessments WHERE submitted_by = ? ORDER BY submitted_at DESC",
        (submitted_by,),
    ).fetchall()
    conn.close()
    return rows


def fetch_by_submitter_as_csv_bytes(submitted_by):
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM assessments WHERE submitted_by = ? ORDER BY submitted_at DESC",
        conn, params=(submitted_by,),
    )
    conn.close()
    return df.to_csv(index=False).encode("utf-8"), len(df)


def fetch_risk_tier_counts():
    """Counts per risk tier across all submitted assessments, for the overview chart."""
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT risk_label FROM assessments", conn)
    conn.close()
    if df.empty:
        return None
    order = ["LOW RISK", "MODERATE RISK", "HIGH RISK"]
    counts = df["risk_label"].value_counts().reindex(order, fill_value=0)
    return counts


def fetch_distinct_patient_labels():
    conn = sqlite3.connect(DB_PATH)
    labels = [r[0] for r in conn.execute(
        "SELECT DISTINCT patient_label FROM assessments ORDER BY patient_label"
    ).fetchall()]
    conn.close()
    return labels


def fetch_probability_history(patient_label):
    """All historical assessments for one patient label, in chronological order --
    lets a doctor see whether a patient's predicted risk is trending up or down
    across multiple submissions, rather than a single point-in-time snapshot."""
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT submitted_at, probability, risk_label FROM assessments "
        "WHERE patient_label = ? ORDER BY submitted_at ASC",
        conn, params=(patient_label,),
    )
    conn.close()
    return df


init_db()

# -------------------------------------------------------------------
# Visual identity
# -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink:        #0F2A2E;
    --teal:       #0E5C63;
    --teal-dark:  #0A454A;
    --canvas:     #F6F8F7;
    --card:       #FFFFFF;
    --line:       #DCE6E4;
    --low:        #1E8E5A;
    --moderate:   #C77A17;
    --high:       #B23A3A;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background-color: var(--canvas); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }

.app-header {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 1.1rem 1.4rem;
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
    border-radius: 14px; margin-bottom: 1.6rem;
}
.app-header .icon { font-size: 2.1rem; line-height: 1; }
.app-header .title { color: #FFFFFF; font-family: 'Space Grotesk', sans-serif;
                      font-weight: 700; font-size: 1.5rem; margin: 0; }
.app-header .subtitle { color: #D7E8E6; font-size: 0.92rem; margin: 0.15rem 0 0 0; }

.section-label {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem;
    color: var(--teal-dark); margin: 0.2rem 0 0.6rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card); border: 1px solid var(--line) !important; border-radius: 12px !important;
}

div.stButton > button {
    background: var(--teal); color: white; font-weight: 600; border-radius: 10px;
    border: none; padding: 0.6rem 1.4rem; width: 100%;
    font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.01em;
}
div.stButton > button:hover { background: var(--teal-dark); color: white; }

.footnote { color: #5B7370; font-size: 0.85rem; }

.info-card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
.info-card h4 {
    font-family: 'Space Grotesk', sans-serif; color: var(--teal-dark);
    margin: 0 0 0.6rem 0; font-size: 1.05rem;
}
.warning-pill {
    display: inline-block; background: #FBEFE4; color: var(--moderate);
    border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.85rem;
    font-weight: 600; margin: 0.2rem 0.3rem 0.2rem 0;
}
.alert-banner {
    background: #FDECEC; border: 1px solid #F3C6C6; border-radius: 12px;
    padding: 1rem 1.3rem; color: var(--high); font-weight: 600; margin-bottom: 1rem;
}
.badge {
    display: inline-block; border-radius: 999px; padding: 0.2rem 0.7rem;
    font-size: 0.78rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# Lower edge of the MODERATE band. Testing this model against the real training
# data showed predicted probabilities are heavily skewed toward 0 (an expected
# result of sepsis being a rare event, ~2% of rows): the vast majority of patients
# score near 0%, confirmed-sepsis rows commonly score anywhere from ~9% up to 100%.
# Using half the HIGH threshold (previously threshold * 0.5, i.e. ~16.7%-33.3%) left
# MODERATE too narrow a band for real predictions to ever land in, so cases jumped
# straight from LOW to HIGH. This fixed, lower edge gives MODERATE a realistic,
# reachable range: any non-trivial risk signal below the confirmed HIGH cutoff.
MODERATE_LOWER_BOUND = 0.05


def risk_gauge_html(probability: float, threshold: float):
    pct = max(0.0, min(1.0, probability)) * 100
    if probability >= threshold:
        color, label = "var(--high)", "HIGH RISK"
    elif probability >= MODERATE_LOWER_BOUND:
        color, label = "var(--moderate)", "MODERATE RISK"
    else:
        color, label = "var(--low)", "LOW RISK"

    html = f"""
    <div style="display:flex; align-items:center; gap:1.6rem;
                background:var(--card); border:1px solid var(--line);
                border-radius:14px; padding:1.2rem 1.6rem;">
      <div style="position:relative; width:118px; height:118px; border-radius:50%;
                  background: conic-gradient({color} {pct}%, #E7EEEC 0);
                  display:flex; align-items:center; justify-content:center; flex-shrink:0;">
        <div style="width:88px; height:88px; border-radius:50%; background:var(--card);
                    display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <div style="font-family:'Space Grotesk',sans-serif; font-size:1.35rem;
                      font-weight:700; color:{color};">{pct:.1f}%</div>
          <div style="font-size:0.68rem; color:#5B7370;">sepsis risk</div>
        </div>
      </div>
      <div>
        <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                    font-size:1.15rem; color:{color};">{label}</div>
        <div style="color:var(--ink); font-size:0.92rem; margin-top:0.25rem;">
          Model estimates a {pct:.1f}% probability of sepsis for this patient.
        </div>
        <div class="footnote" style="margin-top:0.5rem;">
          Decision threshold in use: {threshold:.2f} (selected during Week 7 threshold optimization).
        </div>
      </div>
    </div>
    """
    return html, label, pct


def risk_badge(risk_label):
    color = {"HIGH RISK": "var(--high)", "MODERATE RISK": "var(--moderate)",
             "LOW RISK": "var(--low)"}.get(risk_label, "var(--teal)")
    return f'<span class="badge" style="background:{color}22; color:{color};">{risk_label}</span>'


# -------------------------------------------------------------------
# Session state for login
# -------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None


def render_forgot_password(required_role: str):
    with st.expander("Forgot your password?"):
        method = st.radio(
            "How do you want to reset it?",
            ["Enter my authenticator code", "Email me a reset link"],
            key=f"reset_method_{required_role}",
        )
        all_users = get_all_users()

        if method == "Enter my authenticator code":
            with st.form(f"totp_reset_form_{required_role}"):
                username = st.text_input("Username", key=f"totp_reset_user_{required_role}")
                code = st.text_input("6-digit code from your authenticator app",
                                      key=f"totp_reset_code_{required_role}")
                new_pw = st.text_input("New password", type="password",
                                        key=f"totp_reset_pw_{required_role}")
                go = st.form_submit_button("Verify & set new password")
            if go:
                account = all_users.get(username)
                if not account or account.get("role") != required_role:
                    st.error(f"No {required_role} account found with that username.")
                elif not verify_totp_code(username, code.strip()):
                    st.error("That code is invalid or has expired. Codes refresh every 30 seconds "
                              "-- check your authenticator app and try the latest one.")
                elif len(new_pw) < 8:
                    st.error("Choose a password of at least 8 characters.")
                else:
                    set_account_password(username, new_pw)
                    log_auth_event(username, "PASSWORD_RESET_TOTP", "SUCCESS")
                    st.success("Password updated. You can log in with your new password now.")

        else:  # Email me a reset link
            with st.form(f"email_reset_form_{required_role}"):
                username = st.text_input("Username", key=f"email_reset_user_{required_role}")
                send = st.form_submit_button("Send reset link")
            if send:
                account = all_users.get(username)
                # Same message whether the username exists or not, so this can't be used
                # to probe which usernames are valid accounts.
                generic_msg = ("If that account exists and has an email on file, a reset "
                                "link has been sent to it.")
                if account and account.get("role") == required_role and account.get("email"):
                    token = create_password_reset_token(username)
                    try:
                        base_url = st.secrets["app"]["base_url"].rstrip("/")
                        reset_link = f"{base_url}/?reset_token={token}"
                        ok, msg = send_reset_email(account["email"], reset_link)
                        if not ok:
                            st.error(msg)  # e.g. SMTP not configured -- an admin needs to know this
                        else:
                            st.success(generic_msg)
                    except (KeyError, FileNotFoundError, AttributeError):
                        st.error("This app's own URL isn't configured yet, so reset links can't be "
                                  "built. An administrator needs to add [app] base_url to Secrets.")
                else:
                    st.success(generic_msg)


def render_reset_landing_page():
    """Full-screen 'set a new password' view shown when someone opens a reset-link
    URL (?reset_token=...). Takes over the whole page via st.stop() so nothing else
    renders underneath it."""
    token = st.query_params.get("reset_token")
    if not token:
        return
    st.markdown("""
    <div class="app-header">
        <div class="icon">🔑</div>
        <div><p class="title">Reset your password</p></div>
    </div>
    """, unsafe_allow_html=True)
    username = get_valid_reset_token(token)
    if not username:
        st.error("This reset link is invalid or has expired. Reset links are only valid for "
                  f"{RESET_TOKEN_LIFETIME_MINUTES} minutes and can only be used once -- "
                  "request a new one from the login page.")
        st.stop()
    with st.form("reset_landing_form"):
        new_pw = st.text_input("New password", type="password")
        confirm_pw = st.text_input("Confirm new password", type="password")
        submit = st.form_submit_button("Set new password")
    if submit:
        if len(new_pw) < 8:
            st.error("Choose a password of at least 8 characters.")
        elif new_pw != confirm_pw:
            st.error("Those passwords don't match.")
        else:
            set_account_password(username, new_pw)
            consume_reset_token(token)
            log_auth_event(username, "PASSWORD_RESET_EMAIL", "SUCCESS")
            st.success("Password updated. You can close this tab and log in with your new password.")
    st.stop()


render_reset_landing_page()


def login_form(required_role: str):
    all_users = get_all_users()
    if not all_users:
        st.error("No user accounts are configured. An administrator needs to set up "
                 "accounts in Streamlit secrets before anyone can log in.")
        return
    st.markdown(f"""
    <div class="info-card">
        <h4>🔒 {required_role} Login</h4>
        <p class="footnote">Demo credentials only -- this prototype is not a substitute for
        real hospital-grade authentication.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.form(f"login_form_{required_role}"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        account = all_users.get(username)
        password_ok = account and verify_password(password, account.get("password_hash", ""))
        if password_ok and account["role"] == required_role:
            st.session_state.user = {"username": username, **account}
            log_auth_event(username, "LOGIN", "SUCCESS")
            st.rerun()
        else:
            # Log the attempt whether or not the username exists, without revealing
            # which part (username vs. password vs. role) was wrong.
            log_auth_event(username or "(blank)", "LOGIN", "FAILED")
            st.error(f"Invalid credentials, or this account is not a {required_role} account.")
    render_forgot_password(required_role)


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        'Sepsis Prediction System',
        ['Clinician Dashboard', 'Doctor Review', 'Patient & Family', 'Admin'],
        menu_icon='hospital-fill',
        icons=['clipboard2-pulse', 'stethoscope', 'people-fill', 'shield-lock'],
        default_index=0,
        styles={
            "container": {"background-color": "#F6F8F7"},
            "nav-link-selected": {"background-color": "#0E5C63"},
        },
    )
    st.markdown("---")
    if st.session_state.user:
        st.markdown(f"**Logged in:** {st.session_state.user['name']}")
        st.caption(f"Role: {st.session_state.user['role']}")
        if st.button("Log out"):
            log_auth_event(st.session_state.user["username"], "LOGOUT", "SUCCESS")
            st.session_state.user = None
            st.rerun()

        _me = st.session_state.user["username"]
        _totp_row = get_totp_secret(_me)
        with st.expander("🔑 Authenticator (for password resets)"):
            if _totp_row and _totp_row["enabled"]:
                st.caption("✅ Set up. If you're locked out later, use the code from your "
                           "authenticator app on the login page's 'Forgot your password?' link.")
                if st.button("Replace with a new authenticator", key="totp_redo"):
                    st.session_state["totp_pending_secret"] = start_totp_enrollment(_me)
                    st.rerun()
            else:
                st.caption("Scan this QR code with Google Authenticator, Authy, or similar, "
                           "then enter the 6-digit code it shows to finish setup.")
                if "totp_pending_secret" not in st.session_state:
                    st.session_state["totp_pending_secret"] = start_totp_enrollment(_me)
                pending_secret = st.session_state["totp_pending_secret"]
                st.image(totp_qr_png_bytes(_me, pending_secret))
                st.caption(f"Can't scan? Enter this key manually: `{pending_secret}`")
                confirm_code = st.text_input("Enter the code to confirm setup", key="totp_confirm_code")
                if st.button("Confirm setup", key="totp_confirm_btn"):
                    if confirm_totp_enrollment(_me, confirm_code.strip()):
                        del st.session_state["totp_pending_secret"]
                        st.success("Authenticator set up. You can use it to reset your password anytime.")
                        st.rerun()
                    else:
                        st.error("That code didn't match. Make sure you scanned the QR above and "
                                  "are using the current code.")

# =====================================================================
# CLINICIAN DASHBOARD — assess a patient and send the result to a doctor
# =====================================================================
if selected == 'Clinician Dashboard':

    if not st.session_state.user or not has_permission(st.session_state.user["role"], "submit_assessment"):
        st.markdown("""
        <div class="app-header">
            <div class="icon">🩺</div>
            <div>
                <p class="title">Clinician Dashboard</p>
                <p class="subtitle">Log in to assess a patient and send the result to a doctor.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        login_form("Clinician")

    else:
        st.markdown("""
        <div class="app-header">
            <div class="icon">🩺</div>
            <div>
                <p class="title">Early Sepsis Risk Prediction</p>
                <p class="subtitle">Enter the patient's latest ICU vitals and labs, then predict and send to a doctor.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-label">🏷️ Patient Identifier</div>', unsafe_allow_html=True)
            patient_label = st.text_input(
                "Patient name or ID (as used on your unit)",
                key="patient_label",
                help="Used only to label the submission for the reviewing doctor.",
            )

        grouped_features = set()
        inputs = {}
        for group_name, icon, group_fields in FIELD_GROUPS:
            fields_present = [f for f in group_fields if f in FEATURES]
            if not fields_present:
                continue
            grouped_features.update(fields_present)
            with st.container(border=True):
                st.markdown(f'<div class="section-label">{icon} {group_name}</div>', unsafe_allow_html=True)
                cols = st.columns(4)
                for i, feature in enumerate(fields_present):
                    with cols[i % 4]:
                        inputs[feature] = st.text_input(FIELD_LABELS.get(feature, feature), key=feature)

        leftover = [f for f in FEATURES if f not in grouped_features]
        if leftover:
            with st.container(border=True):
                st.markdown('<div class="section-label">📋 Other Measurements</div>', unsafe_allow_html=True)
                cols = st.columns(4)
                for i, feature in enumerate(leftover):
                    with cols[i % 4]:
                        inputs[feature] = st.text_input(FIELD_LABELS.get(feature, feature), key=feature)

        st.write("")
        predict_clicked = st.button('Predict Sepsis Risk')

        if predict_clicked:
            try:
                user_input_list = [float(inputs[feature]) for feature in FEATURES]
            except ValueError:
                st.error('Please fill in every field with a valid number before predicting.')
            else:
                user_input_arr = np.array(user_input_list).reshape(1, -1)
                sepsis_proba = sepsis_model.predict_proba(user_input_arr)[0][1]
                gauge_html, risk_label, pct = risk_gauge_html(sepsis_proba, THRESHOLD)
                st.session_state["last_result"] = {
                    "gauge_html": gauge_html, "risk_label": risk_label,
                    "probability": float(sepsis_proba), "inputs": inputs,
                }

        if st.session_state.get("last_result"):
            result = st.session_state["last_result"]
            st.write("")
            st.markdown(result["gauge_html"], unsafe_allow_html=True)

            st.write("")
            with st.container(border=True):
                st.markdown('<div class="section-label">✅ Suggested Next Steps</div>', unsafe_allow_html=True)
                for step in CLINICAL_GUIDANCE.get(result["risk_label"], []):
                    st.markdown(f"- {step}")

            st.write("")
            if not st.session_state.get("pending_send"):
                if st.button("📤 Send to Doctor for Review"):
                    st.session_state["pending_send"] = True
                    st.rerun()
            else:
                label = patient_label.strip() if patient_label.strip() else "Unlabeled patient"
                st.markdown(f"""
                <div class="alert-banner" style="background:#FFF6E5; border-color:#F0D9A6; color:var(--moderate);">
                    ⚠️ Confirm before sending: <strong>{label}</strong> — {risk_badge(result['risk_label'])}
                    ({result['probability']*100:.1f}% predicted probability). This will notify the doctor queue.
                </div>
                """, unsafe_allow_html=True)
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("✅ Yes, send it", key="confirm_send"):
                        try:
                            save_assessment(
                                patient_label=label,
                                submitted_by=st.session_state.user["name"],
                                probability=result["probability"],
                                risk_label=result["risk_label"],
                                inputs=result["inputs"],
                            )
                        except Exception as e:
                            st.session_state["pending_send"] = False
                            st.error(
                                f"Couldn't save this assessment: {e}. "
                                "This usually means the app's storage folder isn't writable "
                                "on this deployment -- check with whoever manages the hosting."
                            )
                        else:
                            st.session_state["last_result"] = None
                            st.session_state["pending_send"] = False
                            st.session_state["send_success"] = f"{label} ({result['risk_label']})"
                            st.rerun()
                with cancel_col:
                    if st.button("✖ Cancel", key="cancel_send"):
                        st.session_state["pending_send"] = False
                        st.rerun()

        if st.session_state.get("send_success"):
            st.success(f"Sent to Doctor Review: {st.session_state['send_success']}.")
            st.session_state["send_success"] = None

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-label">📜 My Submissions</div>', unsafe_allow_html=True)
            my_rows = fetch_by_submitter(st.session_state.user["name"])

            if not my_rows:
                st.caption("You haven't sent any assessments to a doctor yet.")
            else:
                my_csv_bytes, my_count = fetch_by_submitter_as_csv_bytes(st.session_state.user["name"])
                st.download_button(
                    f"⬇️ Download my {my_count} submissions as CSV",
                    data=my_csv_bytes,
                    file_name=f"my_sepsis_submissions_{dt.date.today().isoformat()}.csv",
                    mime="text/csv",
                )
                st.write("")
                for row in my_rows:
                    status = "✅ Reviewed" if row["reviewed"] else "⏳ Awaiting review"
                    note_line = f" — Doctor's note: {row['doctor_note']}" if row["reviewed"] and row["doctor_note"] else ""
                    st.markdown(
                        f"{risk_badge(row['risk_label'])} &nbsp; **{row['patient_label']}** "
                        f"&nbsp;·&nbsp; {row['submitted_at']} &nbsp;·&nbsp; {status}{note_line}",
                        unsafe_allow_html=True,
                    )

        st.write("")
        st.markdown(
            '<p class="footnote">This tool supports, but does not replace, clinical judgment. '
            'It is not a substitute for professional medical diagnosis.</p>',
            unsafe_allow_html=True,
        )

# =====================================================================
# DOCTOR REVIEW — see everything clinicians have submitted
# =====================================================================
elif selected == 'Doctor Review':

    if not st.session_state.user or not has_permission(st.session_state.user["role"], "review_assessments"):
        st.markdown("""
        <div class="app-header">
            <div class="icon">🗂️</div>
            <div>
                <p class="title">Doctor Review</p>
                <p class="subtitle">Log in to see patient assessments submitted by clinicians.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        login_form("Doctor")

    else:
        st.markdown("""
        <div class="app-header">
            <div class="icon">🗂️</div>
            <div>
                <p class="title">Doctor Review Queue</p>
                <p class="subtitle">Assessments submitted by clinicians, most recent first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Authentication activity (RBAC: requires view_auth_activity permission) --
        if has_permission(st.session_state.user["role"], "view_auth_activity"):
            with st.container(border=True):
                st.markdown('<div class="section-label">🔐 Recent Authentication Activity</div>',
                            unsafe_allow_html=True)
                st.caption("Login and logout events across all accounts. A full Admin-only "
                           "audit view is planned as a follow-up (see Stage 2).")
                audit_df = fetch_auth_audit(limit=25)
                if audit_df.empty:
                    st.caption("No authentication events recorded yet.")
                else:
                    st.dataframe(audit_df, hide_index=True, width="stretch")

        # --- Overview: risk-tier breakdown across every submission ---------------
        tier_counts = fetch_risk_tier_counts()
        if tier_counts is not None:
            with st.container(border=True):
                st.markdown('<div class="section-label">📊 Overview — Submissions by Risk Tier</div>',
                            unsafe_allow_html=True)
                overview_col, chart_col = st.columns([1, 2])
                with overview_col:
                    st.metric("Total submissions", int(tier_counts.sum()))
                    st.metric("High risk", int(tier_counts["HIGH RISK"]))
                with chart_col:
                    st.bar_chart(tier_counts, color="#0E5C63")

        # --- Per-patient risk trend across multiple submissions -------------------
        patient_labels = fetch_distinct_patient_labels()
        if patient_labels:
            with st.container(border=True):
                st.markdown('<div class="section-label">📈 Patient Risk Trend</div>', unsafe_allow_html=True)
                st.caption("Select a patient to see how their predicted risk has changed across "
                           "multiple submissions -- useful since any single reading is only a snapshot.")
                selected_patient = st.selectbox("Patient", patient_labels, key="trend_patient_select")
                history = fetch_probability_history(selected_patient)
                if len(history) < 2:
                    st.caption(f"Only one submission on record for {selected_patient} so far -- "
                               "a trend will appear once there are multiple assessments for this patient.")
                else:
                    chart_df = history.set_index("submitted_at")[["probability"]]
                    st.line_chart(chart_df, color="#B23A3A")
                    st.caption(f"Decision threshold in use: {THRESHOLD:.2f} (probability above this "
                               "line is classified High Risk).")

        show_only_unreviewed = st.toggle("Show only unreviewed", value=True)
        rows = fetch_assessments(only_unreviewed=show_only_unreviewed)

        csv_bytes, total_count = fetch_all_as_csv_bytes()
        st.download_button(
            f"⬇️ Download all {total_count} records as CSV",
            data=csv_bytes,
            file_name=f"sepsis_assessments_{dt.date.today().isoformat()}.csv",
            mime="text/csv",
        )
        st.caption("Exports every record regardless of the toggle above -- useful as a "
                   "backup, since local storage is cleared on redeploy.")

        if not rows:
            st.info("No assessments to show right now.")
        else:
            for row in rows:
                with st.container(border=True):
                    header_col, badge_col = st.columns([4, 1])
                    with header_col:
                        st.markdown(
                            f"**{row['patient_label']}** &nbsp;·&nbsp; "
                            f"submitted by {row['submitted_by']} &nbsp;·&nbsp; {row['submitted_at']}"
                        )
                    with badge_col:
                        st.markdown(risk_badge(row["risk_label"]), unsafe_allow_html=True)

                    with st.expander("View vitals & labs submitted"):
                        submitted_inputs = json.loads(row["inputs_json"])
                        cols = st.columns(4)
                        for i, (feature, value) in enumerate(submitted_inputs.items()):
                            with cols[i % 4]:
                                st.metric(FIELD_LABELS.get(feature, feature), value)

                    st.progress(min(1.0, row["probability"]),
                               text=f"{row['probability']*100:.1f}% predicted probability of sepsis")

                    if row["reviewed"]:
                        st.caption(f"✅ Reviewed. Note: {row['doctor_note'] or '(none)'}")
                    else:
                        note = st.text_input("Add a note (optional)", key=f"note_{row['id']}")
                        if st.button("Mark as reviewed", key=f"review_{row['id']}"):
                            mark_reviewed(row["id"], note)
                            st.rerun()

# =====================================================================
# PATIENT & FAMILY VIEW — educational + symptom self-check, no ML model
# =====================================================================
elif selected == 'Patient & Family':
    st.markdown("""
    <div class="app-header">
        <div class="icon">💙</div>
        <div>
            <p class="title">Understanding Sepsis</p>
            <p class="subtitle">Information for patients and families — no login needed here.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-banner">
        ⚠️ If you or a loved one suddenly feel much worse, are confused, breathing fast,
        or have a fever with chills — tell a nurse or doctor right away. Don't wait.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🩺 What is sepsis?</h4>
            <p>Sepsis happens when the body's response to an infection injures its own tissues
            and organs. It can develop quickly and become life-threatening, which is why early
            recognition matters so much. The care team uses ongoing monitoring -- including a
            computer-assisted risk tool -- alongside their own clinical judgment to watch for
            early warning signs.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4>👀 Common warning signs</h4>
            <span class="warning-pill">Fever or feeling very cold</span>
            <span class="warning-pill">Fast heart rate</span>
            <span class="warning-pill">Fast breathing</span>
            <span class="warning-pill">Confusion or drowsiness</span>
            <span class="warning-pill">Clammy or sweaty skin</span>
            <span class="warning-pill">Extreme pain or discomfort</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🗣️ What you can do</h4>
            <p>You know your loved one best. If something feels different or wrong -- even if
            you can't quite explain it -- say something to the nursing staff. You are not
            "bothering" anyone by asking questions or raising a concern.</p>
            <p>It's always okay to ask the team directly: <em>"Could this be sepsis? What are
            you watching for?"</em></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4>🤝 How this tool fits in</h4>
            <p>Clinicians use this dashboard as one input among many, then a doctor reviews it.
            It does not replace their exam, judgment, or bedside monitoring. This page
            intentionally does not show a risk score -- that result is only meaningful in the
            care team's hands, alongside everything else they know about your condition.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-label">🔍 Could This Be Sepsis? Self-Check</div>',
                    unsafe_allow_html=True)
        st.caption("Check anything that applies to you or your loved one right now. "
                   "This does not diagnose sepsis -- it helps you decide whether to speak "
                   "with the care team, and how soon.")

        col_a, col_b = st.columns(2)
        with col_a:
            s_fever = st.checkbox("Fever, shivering, or feeling unusually cold")
            s_hr = st.checkbox("Heart racing or pounding")
            s_breath = st.checkbox("Breathing fast, or feeling short of breath")
            s_confusion = st.checkbox("New confusion, disorientation, or unusual sleepiness")
        with col_b:
            s_skin = st.checkbox("Skin that's clammy, sweaty, pale, or blotchy")
            s_urine = st.checkbox("Not urinated in the last 12+ hours")
            s_pain = st.checkbox("Extreme pain or severe discomfort")
            s_worried = st.checkbox("A strong feeling that something is seriously wrong")

        check_clicked = st.button("Check Symptoms")

        if check_clicked:
            symptoms = {
                "fever": s_fever, "heart rate": s_hr, "breathing": s_breath,
                "confusion": s_confusion, "skin": s_skin, "urine": s_urine,
                "pain": s_pain, "worried": s_worried,
            }
            checked_count = sum(symptoms.values())
            critical_flags = symptoms["confusion"] or symptoms["urine"] or symptoms["worried"]

            st.write("")
            if checked_count == 0:
                st.markdown("""
                <div class="info-card" style="border-left: 4px solid var(--low);">
                    <h4 style="color:var(--low);">🟢 No urgent signs selected</h4>
                    <p>Based on what you selected, nothing here signals an emergency right now.
                    Keep watching how you or your loved one feels, and don't hesitate to check
                    in with the care team if anything changes.</p>
                </div>
                """, unsafe_allow_html=True)
            elif critical_flags or checked_count >= 3:
                st.markdown("""
                <div class="alert-banner">
                    🔴 These signs need attention now — please alert a nurse or doctor
                    immediately, or call emergency services if you're not currently in a
                    care facility.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card" style="border-left: 4px solid var(--moderate);">
                    <h4 style="color:var(--moderate);">🟡 Worth telling the care team soon</h4>
                    <p>What you selected is worth flagging to a nurse or doctor soon so they
                    can take a closer look, even if it doesn't feel like an emergency yet.</p>
                </div>
                """, unsafe_allow_html=True)

            st.caption("This self-check is a general awareness tool, not a diagnosis. "
                       "When in doubt, it's always okay to ask the care team directly.")

    st.write("")
    st.markdown(
        '<p class="footnote">This page provides general information only and is not a '
        'substitute for professional medical advice, diagnosis, or treatment. If you have '
        'concerns about your or a loved one\'s condition, speak with a member of your care team.</p>',
        unsafe_allow_html=True,
    )

# =====================================================================
# ADMIN VIEW — create accounts and reset passwords. This is what actually makes
# account creation and password recovery possible: Streamlit secrets cannot be
# edited from inside the running app, so any in-app account management has to be
# backed by the database instead (see get_all_users() above).
# =====================================================================
else:
    if not st.session_state.user or not has_permission(st.session_state.user["role"], "manage_accounts"):
        st.markdown("""
        <div class="app-header">
            <div class="icon">🛡️</div>
            <div>
                <p class="title">Admin</p>
                <p class="subtitle">Log in to create accounts or reset passwords.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        login_form("Admin")

    else:
        st.markdown("""
        <div class="app-header">
            <div class="icon">🛡️</div>
            <div>
                <p class="title">Account Administration</p>
                <p class="subtitle">Create a new account, or reset an existing one's password.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-label">➕ Create or Reset an Account</div>', unsafe_allow_html=True)
            st.caption("Using a username that already exists resets that account's password "
                       "and updates its name/role, rather than creating a duplicate.")
            with st.form("admin_account_form", clear_on_submit=True):
                acc_username = st.text_input("Username", key="admin_acc_username")
                acc_name = st.text_input("Full name", key="admin_acc_name")
                acc_email = st.text_input("Email (needed for 'email me a reset link')", key="admin_acc_email")
                acc_role = st.selectbox("Role", ["Clinician", "Doctor", "Admin"], key="admin_acc_role")
                acc_password = st.text_input("Password (min. 8 characters)", type="password", key="admin_acc_password")
                acc_password_confirm = st.text_input("Confirm password", type="password", key="admin_acc_password_confirm")
                acc_submitted = st.form_submit_button("Save Account")

            if acc_submitted:
                existing_users = get_all_users()
                is_reset = acc_username in existing_users
                if not acc_username or not acc_name or not acc_password:
                    st.error("Username, full name, and password are all required.")
                elif len(acc_password) < 8:
                    st.error("Password must be at least 8 characters.")
                elif acc_password != acc_password_confirm:
                    st.error("Passwords do not match.")
                else:
                    create_or_reset_account(acc_username, acc_password, acc_role, acc_name,
                                             email=acc_email.strip() or None)
                    log_auth_event(
                        st.session_state.user["username"],
                        "PASSWORD_RESET" if is_reset else "ACCOUNT_CREATED",
                        "SUCCESS",
                    )
                    action_word = "reset" if is_reset else "created"
                    st.success(f"Account for {acc_name} ({acc_username}) was {action_word} successfully.")

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-label">👥 All Accounts</div>', unsafe_allow_html=True)
            accounts = fetch_all_accounts_overview()
            if not accounts:
                st.caption("No accounts found.")
            else:
                import pandas as pd
                st.dataframe(pd.DataFrame(accounts), hide_index=True, width="stretch")

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-label">🔐 Recent Authentication Activity</div>', unsafe_allow_html=True)
            audit_df = fetch_auth_audit(limit=25)
            if audit_df.empty:
                st.caption("No authentication events recorded yet.")
            else:
                st.dataframe(audit_df, hide_index=True, width="stretch")

        st.write("")
        st.markdown(
            '<p class="footnote">Account data is stored in the app\'s local database, which is '
            'cleared when the app redeploys or sleeps from inactivity (see the deployment notes '
            'in the User Guide). Re-create any accounts made here if that happens.</p>',
            unsafe_allow_html=True,
        )
