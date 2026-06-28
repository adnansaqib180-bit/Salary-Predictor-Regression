import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
)

# ─── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("REG_model.pkl", "rb") as f:          # ← MODEL FILE NAME YAHAN DAALEIN
        model = pickle.load(f)
    with open("scaller.pkl", "rb") as f:          # ← SCALER FILE NAME YAHAN DAALEIN
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid #2a3550;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 700; color: #e2e8f0; margin: 0 0 0.4rem; }
    .hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 0.6rem;
        padding-left: 0.1rem;
    }

    .card {
        background: #1e2533;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    .result-low  { background: linear-gradient(135deg,#052e16,#14532d); border:1px solid #16a34a; border-radius:12px; padding:1.6rem 2rem; text-align:center; }
    .result-high { background: linear-gradient(135deg,#2d0a0a,#7f1d1d); border:1px solid #ef4444; border-radius:12px; padding:1.6rem 2rem; text-align:center; }
    .result-low  h2 { color:#4ade80; font-size:1.6rem; margin:0 0 0.3rem; }
    .result-high h2 { color:#f87171; font-size:1.6rem; margin:0 0 0.3rem; }
    .result-low  p  { color:#86efac; margin:0; }
    .result-high p  { color:#fca5a5; margin:0; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }

    label { color: #cbd5e1 !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>👥 Employee Attrition Predictor</h1>
    <p>Fill in the employee details below to predict whether they are likely to leave the company.</p>
</div>
""", unsafe_allow_html=True)

# ─── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Section 1: Personal Info ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                  format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])

    # ── Section 2: Compensation ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Compensation</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        daily_rate   = st.number_input("Daily Rate",   min_value=100,   max_value=1500,  value=800)
    with col2:
        hourly_rate  = st.number_input("Hourly Rate",  min_value=30,    max_value=100,   value=65)
    with col3:
        monthly_rate = st.number_input("Monthly Rate", min_value=2000,  max_value=27000, value=14000)

    # ── Section 3: Job Details ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Job Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        job_level        = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_involvement  = st.selectbox("Job Involvement", [1, 2, 3, 4],
                                         format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    with col2:
        overtime         = st.selectbox("OverTime", ["Yes", "No"])
        performance      = st.selectbox("Performance Rating", [1, 2, 3, 4],
                                         format_func=lambda x: {1:"Low",2:"Good",3:"Excellent",4:"Outstanding"}[x])
    with col3:
        department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
        job_role   = st.selectbox("Job Role", [
            "Healthcare Representative", "Human Resources", "Laboratory Technician",
            "Manager", "Manufacturing Director", "Research Director",
            "Research Scientist", "Sales Executive", "Sales Representative"
        ])

    # ── Section 4: Education Field ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Education Field</div>', unsafe_allow_html=True)
    edu_field = st.selectbox("Field of Education", [
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"
    ])

    # ── Section 5: Experience ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Work Experience</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        num_companies       = st.number_input("Num Companies Worked",     min_value=0, max_value=9,  value=2)
        total_working_years = st.number_input("Total Working Years",      min_value=0, max_value=40, value=10)
    with col2:
        training_times      = st.number_input("Training Times Last Year", min_value=0, max_value=6,  value=2)
        years_at_company    = st.number_input("Years At Company",         min_value=0, max_value=40, value=5)
    with col3:
        years_since_promo   = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)

    submitted = st.form_submit_button("🔍  Predict Attrition")

# ─── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    gender_enc     = 1 if gender == "Male" else 0
    overtime_enc   = 1 if overtime == "Yes" else 0

    dep_rd    = 1 if department == "Research & Development" else 0
    dep_sales = 1 if department == "Sales" else 0

    field_map = {
        "Life Sciences":    [1,0,0,0,0],
        "Marketing":        [0,1,0,0,0],
        "Medical":          [0,0,1,0,0],
        "Other":            [0,0,0,1,0],
        "Technical Degree": [0,0,0,0,1],
        "Human Resources":  [0,0,0,0,0],
    }
    f_ls, f_mkt, f_med, f_oth, f_tech = field_map[edu_field]

    role_map = {
        "Human Resources":          [1,0,0,0,0,0,0,0],
        "Laboratory Technician":    [0,1,0,0,0,0,0,0],
        "Manager":                  [0,0,1,0,0,0,0,0],
        "Manufacturing Director":   [0,0,0,1,0,0,0,0],
        "Research Director":        [0,0,0,0,1,0,0,0],
        "Research Scientist":       [0,0,0,0,0,1,0,0],
        "Sales Executive":          [0,0,0,0,0,0,1,0],
        "Sales Representative":     [0,0,0,0,0,0,0,1],
        "Healthcare Representative":[0,0,0,0,0,0,0,0],
    }
    r_hr,r_lt,r_mgr,r_md,r_rd,r_rs,r_se,r_srep = role_map[job_role]

    # Column order must match training data exactly
    feature_names = [
        "Age","Attrition","DailyRate","Education","Gender","HourlyRate",
        "JobInvolvement","JobLevel","MonthlyRate","NumCompaniesWorked",
        "OverTime","PerformanceRating","TotalWorkingYears","TrainingTimesLastYear",
        "YearsAtCompany","YearsSinceLastPromotion",
        "dep_Research & Development","dep_Sales",
        "feild_Life Sciences","feild_Marketing","feild_Medical","feild_Other","feild_Technical Degree",
        "role_Human Resources","role_Laboratory Technician","role_Manager",
        "role_Manufacturing Director","role_Research Director","role_Research Scientist",
        "role_Sales Executive","role_Sales Representative"
    ]

    raw_values = [
        age, 0, daily_rate, education, gender_enc, hourly_rate,
        job_involvement, job_level, monthly_rate, num_companies,
        overtime_enc, performance, total_working_years, training_times,
        years_at_company, years_since_promo,
        dep_rd, dep_sales,
        f_ls, f_mkt, f_med, f_oth, f_tech,
        r_hr, r_lt, r_mgr, r_md, r_rd, r_rs, r_se, r_srep
    ]

    input_df = pd.DataFrame([raw_values], columns=feature_names)

    # Drop target column before scaling/predicting
    X = input_df.drop(columns=["Attrition"])

    X_scaled    = scaler.transform(X)
    prediction  = model.predict(X_scaled)[0]
    proba       = model.predict_proba(X_scaled)[0]
    risk_pct    = round(proba[1] * 100, 1)

    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <h2>⚠️ High Attrition Risk</h2>
            <p>This employee has a <strong>{risk_pct}%</strong> probability of leaving the company.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <h2>✅ Low Attrition Risk</h2>
            <p>This employee has only a <strong>{risk_pct}%</strong> probability of leaving the company.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("Stay Probability",  f"{round(proba[0]*100,1)}%")
    col_b.metric("Leave Probability", f"{risk_pct}%")