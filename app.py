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

# Load the feature list + classification threshold documented in Week 8.
# Falls back to sensible defaults if the documentation file isn't found,
# so the app still runs even if only the .pkl was copied over.
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

# Human-friendly labels for each ICU feature the model expects
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

# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
with st.sidebar:
    selected = option_menu('Sepsis Prediction System',
                            ['Sepsis Risk Prediction'],
                            menu_icon='hospital-fill',
                            icons=['activity'],
                            default_index=0)

# -------------------------------------------------------------------
# Sepsis Risk Prediction Page
# -------------------------------------------------------------------
if selected == 'Sepsis Risk Prediction':

    # page title
    st.title('Early Sepsis Risk Prediction using ML')
    st.caption('Enter the patient\'s latest ICU vitals and labs below, then click '
               '"Predict Sepsis Risk" to get a risk score.')

    # getting the input data from the user, 4 fields per row
    inputs = {}
    cols = st.columns(4)
    for i, feature in enumerate(FEATURES):
        with cols[i % 4]:
            inputs[feature] = st.text_input(FIELD_LABELS.get(feature, feature))

    # code for Prediction
    sepsis_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict Sepsis Risk'):
        try:
            user_input = [float(inputs[feature]) for feature in FEATURES]
        except ValueError:
            st.error('Please fill in every field with a valid number before predicting.')
        else:
            user_input = np.array(user_input).reshape(1, -1)
            sepsis_proba = sepsis_model.predict_proba(user_input)[0][1]
            sepsis_prediction = int(sepsis_proba >= THRESHOLD)

            if sepsis_prediction == 1:
                sepsis_diagnosis = (f'⚠️ HIGH RISK — the model estimates a '
                                     f'{sepsis_proba * 100:.1f}% probability of sepsis.')
                st.error(sepsis_diagnosis)
            elif sepsis_proba >= THRESHOLD * 0.5:
                sepsis_diagnosis = (f'🟡 MODERATE RISK — the model estimates a '
                                     f'{sepsis_proba * 100:.1f}% probability of sepsis. '
                                     f'Continue close monitoring.')
                st.warning(sepsis_diagnosis)
            else:
                sepsis_diagnosis = (f'✅ LOW RISK — the model estimates a '
                                     f'{sepsis_proba * 100:.1f}% probability of sepsis.')
                st.success(sepsis_diagnosis)

            st.caption(f'Decision threshold in use: {THRESHOLD:.2f} '
                       '(selected during Week 7 threshold optimization).')

    st.divider()
    st.caption('This tool supports, but does not replace, clinical judgment. '
               'It is not a substitute for professional medical diagnosis.')
