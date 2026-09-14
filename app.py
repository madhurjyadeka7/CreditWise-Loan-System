import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model & preprocessors
# -----------------------------
ohe = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CreditWise Loan System",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<div class="main-title">💳 CreditWise Loan System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Loan Approval Prediction using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Applicant Information
# -----------------------------
st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    applicant_id = st.number_input(
        "Applicant ID",
        min_value=1,
        value=1001,
        step=1
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

with col2:
    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=680.0,
        step=1.0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01
    )

with col3:
    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=25000.0,
        step=500.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=20000.0,
        step=500.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        max_value=120,
        value=48,
        step=1
    )


st.divider()


# -----------------------------
# Categorical Information
# -----------------------------
st.header("📋 Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

with col2:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Personal", "Car", "Business", "Education"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

with col3:
    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Private", "Government", "MNC", "Unemployed"]
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check Loan Eligibility", use_container_width=True):

    # Education_Level was LabelEncoded in the notebook.
    # With sklearn LabelEncoder:
    # Graduate -> 0
    # Not Graduate -> 1
    education_encoded = {
        "Graduate": 0,
        "Not Graduate": 1
    }[education_level]

    # Create categorical dataframe
    cat_data = pd.DataFrame({
        "Employment_Status": [employment_status],
        "Marital_Status": [marital_status],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Gender": [gender],
        "Employer_Category": [employer_category]
    })

    # Apply the same OneHotEncoder used during training
    encoded = ohe.transform(cat_data)

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out([
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ])
    )

    # Base numeric/features
    input_df = pd.DataFrame({
        "Applicant_ID": [applicant_id],
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Age": [age],
        "Dependents": [dependents],
        "Existing_Loans": [existing_loans],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Education_Level": [education_encoded],
        "DTI_Ratio_sq": [dti_ratio ** 2],
        "Credit_Score_sq": [credit_score ** 2],
        "Applicant_Income_log": [np.log1p(applicant_income)]
    })

    # Combine numeric + encoded categorical features
    final_input = pd.concat(
        [input_df, encoded_df],
        axis=1
    )

    # Make the exact same column order used by the trained scaler
    expected_columns = list(scaler.feature_names_in_)

    for column in expected_columns:
        if column not in final_input.columns:
            final_input[column] = 0

    final_input = final_input[expected_columns]

    # Scale input
    final_scaled = scaler.transform(final_input)

    # Prediction
    prediction = model.predict(final_scaled)[0]

    st.divider()
    st.header("📊 Prediction Result")

    if prediction == 1:
        st.markdown(
            '<div class="result-box">✅ Loan Approved</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box">❌ Loan Not Approved</div>',
            unsafe_allow_html=True
        )

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(final_scaled)[0]
        approval_probability = probability[1] * 100

        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )

st.caption(
    "CreditWise Loan System | Powered by Gaussian Naive Bayes"
)