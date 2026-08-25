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

# getting the working directory of app.py
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

# Clinical groupings a bedside user actually thinks in. Any feature not listed
# here (e.g. if the model was retrained on a different feature set) falls
# through to "Other Measurements" automatically, so the app never breaks.
FIELD_GROUPS = [
    ("Vital Signs", "🫀", ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']),
    ("Labs & Blood Gas", "🧪", ['BaseExcess', 'FiO2', 'pH', 'Glucose', 'Potassium', 'Hct']),
    ("Patient & Admission Info", "🧍", ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'Hour', 'ICULOS']),
]

# -------------------------------------------------------------------
# Visual identity — clinical, calm, legible. Deep teal for trust/clinical
# authority, a soft off-white canvas, and a restrained three-color risk
# scale (the one place the palette is allowed to speak loudly).
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

/* App header */
.app-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 1.4rem;
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
    border-radius: 14px;
    margin-bottom: 1.6rem;
}
.app-header .icon { font-size: 2.1rem; line-height: 1; }
.app-header .title { color: #FFFFFF; font-family: 'Space Grotesk', sans-serif;
                      font-weight: 700; font-size: 1.5rem; margin: 0; }
.app-header .subtitle { color: #D7E8E6; font-size: 0.92rem; margin: 0.15rem 0 0 0; }

/* Section headers inside the form */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--teal-dark);
    margin: 0.2rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Bordered containers used for each clinical group */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
}

/* Predict button */
div.stButton > button {
    background: var(--teal);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.4rem;
    width: 100%;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.01em;
}
div.stButton > button:hover { background: var(--teal-dark); color: white; }

.footnote { color: #5B7370; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


def risk_gauge_html(probability: float, threshold: float) -> str:
    """Render a compact conic-gradient risk gauge as raw HTML (no extra chart library needed)."""
    pct = max(0.0, min(1.0, probability)) * 100
    if probability >= threshold:
        color, label = "var(--high)", "HIGH RISK"
    elif probability >= threshold * 0.5:
        color, label = "var(--moderate)", "MODERATE RISK"
    else:
        color, label = "var(--low)", "LOW RISK"

    return f"""
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


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
with st.sidebar:
    selected = option_menu('Sepsis Prediction System',
                            ['Sepsis Risk Prediction'],
                            menu_icon='hospital-fill',
                            icons=['activity'],
                            default_index=0,
                            styles={
                                "container": {"background-color": "#F6F8F7"},
                                "nav-link-selected": {"background-color": "#0E5C63"},
                            })
    st.markdown("---")
    st.markdown("**About this tool**")
    st.caption("Estimates early sepsis risk from the latest ICU vitals and labs, "
               "trained on the PhysioNet 2019 Sepsis Challenge dataset.")
    st.markdown("**Risk levels**")
    st.markdown(
        "🟢 Low &nbsp;·&nbsp; 🟡 Moderate &nbsp;·&nbsp; 🔴 High",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Sepsis Risk Prediction Page
# -------------------------------------------------------------------
if selected == 'Sepsis Risk Prediction':

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
            st.markdown(f'<div class="section-label">{icon} {group_name}</div>',
                        unsafe_allow_html=True)
            cols = st.columns(4)
            for i, feature in enumerate(fields_present):
                with cols[i % 4]:
                    inputs[feature] = st.text_input(FIELD_LABELS.get(feature, feature),
                                                     key=feature)

    # Any feature not covered by the known clinical groups still gets a field,
    # so the app never silently drops an input the model actually needs.
    leftover = [f for f in FEATURES if f not in grouped_features]
    if leftover:
        with st.container(border=True):
            st.markdown('<div class="section-label">📋 Other Measurements</div>',
                        unsafe_allow_html=True)
            cols = st.columns(4)
            for i, feature in enumerate(leftover):
                with cols[i % 4]:
                    inputs[feature] = st.text_input(FIELD_LABELS.get(feature, feature),
                                                     key=feature)

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
            st.write("")
            st.markdown(risk_gauge_html(sepsis_proba, THRESHOLD), unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<p class="footnote">This tool supports, but does not replace, clinical judgment. '
        'It is not a substitute for professional medical diagnosis.</p>',
        unsafe_allow_html=True,
    )
