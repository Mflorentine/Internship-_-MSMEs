import os
import json
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

# Plain-language, risk-tier-specific next steps for the clinician view. Phrased as suggestions
# that support, not replace, clinical judgment and local sepsis protocols (e.g. Surviving
# Sepsis Campaign bundles).
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
</style>
""", unsafe_allow_html=True)


def risk_gauge_html(probability: float, threshold: float):
    """Render a compact conic-gradient risk gauge as raw HTML. Returns (html, label)."""
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
    return html, label


# -------------------------------------------------------------------
# Sidebar navigation — role-based views
# -------------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        'Sepsis Prediction System',
        ['Clinician / Doctor', 'Patient & Family'],
        menu_icon='hospital-fill',
        icons=['clipboard2-pulse', 'people-fill'],
        default_index=0,
        styles={
            "container": {"background-color": "#F6F8F7"},
            "nav-link-selected": {"background-color": "#0E5C63"},
        },
    )
    st.markdown("---")
    if selected == 'Clinician / Doctor':
        st.markdown("**About this view**")
        st.caption("Full risk-assessment dashboard for care team use: enter the patient's "
                   "latest vitals and labs to get a risk score and suggested next steps.")
        st.markdown("**Risk levels**")
        st.markdown("🟢 Low &nbsp;·&nbsp; 🟡 Moderate &nbsp;·&nbsp; 🔴 High", unsafe_allow_html=True)
    else:
        st.markdown("**About this view**")
        st.caption("Plain-language information about sepsis for patients and families. "
                   "This page does not run the prediction model or show a risk score.")

# =====================================================================
# CLINICIAN / DOCTOR VIEW
# =====================================================================
if selected == 'Clinician / Doctor':

    st.markdown("""
    <div class="app-header">
        <div class="icon">🩺</div>
        <div>
            <p class="title">Early Sepsis Risk Prediction</p>
            <p class="subtitle">Enter the patient's latest ICU vitals and labs, then predict.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            user_input = [float(inputs[feature]) for feature in FEATURES]
        except ValueError:
            st.error('Please fill in every field with a valid number before predicting.')
        else:
            user_input = np.array(user_input).reshape(1, -1)
            sepsis_proba = sepsis_model.predict_proba(user_input)[0][1]
            gauge_html, risk_label = risk_gauge_html(sepsis_proba, THRESHOLD)
            st.write("")
            st.markdown(gauge_html, unsafe_allow_html=True)

            st.write("")
            with st.container(border=True):
                st.markdown('<div class="section-label">✅ Suggested Next Steps</div>', unsafe_allow_html=True)
                for step in CLINICAL_GUIDANCE.get(risk_label, []):
                    st.markdown(f"- {step}")

    st.write("")
    st.markdown(
        '<p class="footnote">This tool supports, but does not replace, clinical judgment. '
        'It is not a substitute for professional medical diagnosis.</p>',
        unsafe_allow_html=True,
    )

# =====================================================================
# PATIENT & FAMILY VIEW — educational only, no data entry, no score
# =====================================================================
else:
    st.markdown("""
    <div class="app-header">
        <div class="icon">💙</div>
        <div>
            <p class="title">Understanding Sepsis</p>
            <p class="subtitle">Information for patients and families — no data entry needed here.</p>
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
            recognition matters so much. The care team here uses ongoing monitoring — including
            a computer-assisted risk tool — alongside their own clinical judgment to watch for
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
            <p>You know your loved one best. If something feels different or wrong — even if
            you can't quite explain it — say something to the nursing staff. You are not
            "bothering" anyone by asking questions or raising a concern.</p>
            <p>It's always okay to ask the team directly: <em>"Could this be sepsis? What are
            you watching for?"</em></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>🤝 How this tool fits in</h4>
            <p>This dashboard is used by clinicians and doctors as one input among many. It
            does not replace their exam, judgment, or the monitoring equipment at the bedside.
            This page intentionally does not show a risk score — that result is only meaningful
            in the hands of your care team, alongside everything else they know about your
            condition.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<p class="footnote">This page provides general information only and is not a '
        'substitute for professional medical advice, diagnosis, or treatment. If you have '
        'concerns about your or a loved one\'s condition, speak with a member of your care team.</p>',
        unsafe_allow_html=True,
    )
