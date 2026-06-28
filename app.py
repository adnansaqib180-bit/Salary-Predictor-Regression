import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide",
)

# ─── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("REG_model.pkl")   # ← MODEL FILE NAME YAHAN DAALEIN
    scaler = joblib.load("scaller.pkl")   # ← SCALER FILE NAME YAHAN DAALEIN
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
    .result-box {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .result-box h2 { color: #4ade80; font-size: 2.4rem; margin: 0 0 0.3rem; }
    .result-box p  { color: #86efac; margin: 0; font-size: 1rem; }
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
    <h1>💰 Employee Salary Predictor</h1>
    <p>Employee ki details fill karo aur predicted salary dekho.</p>
</div>
""", unsafe_allow_html=True)

# ─── Input Form ────────────────────────────────────────────────────────────────
with st.form("salary_form"):

    # ── Personal Info ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                  format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])

    # ── Compensation ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Compensation</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        daily_rate   = st.number_input("Daily Rate",   min_value=100,   max_value=1500,  value=800)
    with col2:
        hourly_rate  = st.number_input("Hourly Rate",  min_value=30,    max_value=100,   value=65)
    with col3:
        monthly_rate = st.number_input("Monthly Rate", min_value=2000,  max_value=27000, value=14000)

    # ── Job Details ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Job Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        job_level       = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4],
                                        format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    with col2:
        overtime    = st.selectbox("OverTime", ["Yes", "No"])
        performance = st.selectbox("Performance Rating", [1, 2, 3, 4],
                                    format_func=lambda x: {1:"Low",2:"Good",3:"Excellent",4:"Outstanding"}[x])
    with col3:
        attrition  = st.selectbox("Attrition", ["No", "Yes"])
        department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])

    job_role = st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ])

    # ── Education Field ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Education Field</div>', unsafe_allow_html=True)
    edu_field = st.selectbox("Field of Education", [
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"
    ])

    # ── Experience ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Work Experience</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        num_companies       = st.number_input("Num Companies Worked",      min_value=0, max_value=9,  value=2)
        total_working_years = st.number_input("Total Working Years",       min_value=0, max_value=40, value=10)
    with col2:
        training_times   = st.number_input("Training Times Last Year",  min_value=0, max_value=6,  value=2)
        years_at_company = st.number_input("Years At Company",          min_value=0, max_value=40, value=5)
    with col3:
        years_since_promo = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)

    submitted = st.form_submit_button("💰  Predict Salary")

# ─── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    gender_enc    = 1 if gender == "Male" else 0
    overtime_enc  = 1 if overtime == "Yes" else 0
    attrition_enc = 1 if attrition == "Yes" else 0

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

    feature_names = [
        "Age", "Attrition", "DailyRate", "Education", "Gender", "HourlyRate",
        "JobInvolvement", "JobLevel", "MonthlyRate", "NumCompaniesWorked",
        "OverTime", "PerformanceRating", "TotalWorkingYears", "TrainingTimesLastYear",
        "YearsAtCompany", "YearsSinceLastPromotion",
        "dep_Research & Development", "dep_Sales",
        "feild_Life Sciences", "feild_Marketing", "feild_Medical", "feild_Other",
        "feild_Technical Degree", "role_Human Resources", "role_Laboratory Technician",
        "role_Manager", "role_Manufacturing Director", "role_Research Director",
        "role_Research Scientist", "role_Sales Executive", "role_Sales Representative"
    ]

    raw_values = [
        age, attrition_enc, daily_rate, education, gender_enc, hourly_rate,
        job_involvement, job_level, monthly_rate, num_companies,
        overtime_enc, performance, total_working_years, training_times,
        years_at_company, years_since_promo,
        dep_rd, dep_sales,
        f_ls, f_mkt, f_med, f_oth, f_tech,
        r_hr, r_lt, r_mgr, r_md, r_rd, r_rs, r_se, r_srep
    ]

    X = pd.DataFrame([raw_values], columns=feature_names)

    # Scaler sirf inhi 12 columns pe fit hua tha
    scale_cols = [
        "Age", "DailyRate", "Education", "HourlyRate",
        "JobInvolvement", "JobLevel", "MonthlyRate",
        "NumCompaniesWorked", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsSinceLastPromotion"
    ]
    other_cols    = [c for c in X.columns if c not in scale_cols]
    X_scaled_part = scaler.transform(X[scale_cols])
    X_other_part  = X[other_cols].values
    X_final       = np.hstack([X_scaled_part, X_other_part])

    predicted_salary = model.predict(X_final)[0]

    st.markdown("---")
    st.markdown(f"""
    <div class="result-box">
        <h2>PKR {predicted_salary:,.0f}</h2>
        <p>Predicted Monthly Salary based on employee profile</p>
    </div>
    """, unsafe_allow_html=True)