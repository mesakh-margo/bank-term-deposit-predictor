# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('logreg_pipeline.joblib')

pipeline = load_pipeline()

st.title("Bank Marketing Subscription Predictor")

st.write("""
Enter customer information to predict the probability of subscribing to a term deposit.
""")

# --- Input fields (customize as needed) ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job_type = st.selectbox("Job Type", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
])
marital_status = st.selectbox("Marital Status", [
    "divorced", "married", "single", "unknown"
])
education = st.selectbox("Education", [
    "primary", "secondary", "tertiary", "unknown"
])
credit_default_flag = st.selectbox("Credit Default", ["yes", "no", "unknown"])
avg_yearly_balance = st.number_input("Average Yearly Balance (EUR)", value=1000)
housing_loan_flag = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
personal_loan_flag = st.selectbox("Personal Loan", ["yes", "no", "unknown"])
contact_type = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
last_contact_day = st.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])
last_contact_month = st.selectbox("Last Contact Month", [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
])
campaign_contact_count = st.number_input("Campaign Contact Count", min_value=1, value=1)
days_since_prev_contact = st.number_input("Days Since Previous Contact", min_value=-1, value=-1)
previous_contacts_count = st.number_input("Previous Contacts Count", min_value=0, value=0)
previous_campaign_outcome = st.selectbox("Previous Campaign Outcome", [
    "failure", "nonexistent", "success"
])

# --- Collect inputs into a DataFrame ---
input_dict = {
    "age": [age],
    "job_type": [job_type],
    "marital_status": [marital_status],
    "education": [education],
    "credit_default_flag": [credit_default_flag],
    "avg_yearly_balance": [avg_yearly_balance],
    "housing_loan_flag": [housing_loan_flag],
    "personal_loan_flag": [personal_loan_flag],
    "contact_type": [contact_type],
    "last_contact_day": [last_contact_day],
    "last_contact_month": [last_contact_month],
    "campaign_contact_count": [campaign_contact_count],
    "days_since_prev_contact": [days_since_prev_contact],
    "previous_contacts_count": [previous_contacts_count],
    "previous_campaign_outcome": [previous_campaign_outcome]
}

input_df = pd.DataFrame(input_dict)

# --- Prediction ---
if st.button("Predict"):
    pred_proba = pipeline.predict_proba(input_df)[0, 1]
    pred_class = pipeline.predict(input_df)[0]
    st.write(f"**Probability of Subscription:** {pred_proba:.2f}")
    st.write(f"**Prediction:** {'Subscribed' if pred_class == 'yes' or pred_class == 1 else 'Not Subscribed'}")