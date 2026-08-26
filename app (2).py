import os
import json
import sqlite3
import datetime as dt

import joblib
import numpy as np
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
# DEMO CREDENTIALS -- for prototype / internship demonstration purposes only.
# Plain-text, hardcoded accounts are NOT appropriate for a real deployment
# handling real patient data. A production version needs proper authentication
# (e.g. hashed passwords via `streamlit-authenticator`, hospital SSO/OAuth),
# encrypted storage, and an audit trail -- flag this clearly in your report.
# ---------------------------------------------------------------------------
USERS = {
    "clinician1": {"password": "clinician123", "role": "Clinician", "name": "Nurse A. Uwase"},
    "clinician2": {"password": "clinician123", "role": "Clinician", "name": "Nurse B. Mugisha"},
    "doctor1":    {"password": "doctor123",    "role": "Doctor",    "name": "Dr. C. Habimana"},
}

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
    conn.commit()
    conn.close()


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


def risk_gauge_html(probability: float, threshold: float):
    pct = max(0.0, min(1.0, probability)) * 100
    if probability >= threshold:
        color, label = "var(--high)", "HIGH RISK"
    elif probability >= threshold * 0.5:
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


def login_form(required_role: str):
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
        account = USERS.get(username)
        if account and account["password"] == password and account["role"] == required_role:
            st.session_state.user = {"username": username, **account}
            st.rerun()
        else:
            st.error(f"Invalid credentials, or this account is not a {required_role} account.")


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        'Sepsis Prediction System',
        ['Clinician Dashboard', 'Doctor Review', 'Patient & Family'],
        menu_icon='hospital-fill',
        icons=['clipboard2-pulse', 'stethoscope', 'people-fill'],
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
            st.session_state.user = None
            st.rerun()

# =====================================================================
# CLINICIAN DASHBOARD — assess a patient and send the result to a doctor
# =====================================================================
if selected == 'Clinician Dashboard':

    if not st.session_state.user or st.session_state.user["role"] != "Clinician":
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
                        save_assessment(
                            patient_label=label,
                            submitted_by=st.session_state.user["name"],
                            probability=result["probability"],
                            risk_label=result["risk_label"],
                            inputs=result["inputs"],
                        )
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

    if not st.session_state.user or st.session_state.user["role"] != "Doctor":
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
else:
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
