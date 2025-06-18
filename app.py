import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<p class="main-header">Bank Term Deposit Prediction</p>', unsafe_allow_html=True)
st.markdown("Predict whether a client will subscribe to a term deposit based on various features.")

# Add this function definition before loading the model
def clip_values(x):
    """Clips values in a numpy array."""
    return np.clip(x, a_min=-0.999, a_max=None)

# Load model and preprocessor
@st.cache_resource
def load_pipeline():
    try:
        with open('final_pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline, None
    except Exception as e:
        return None, str(e)

pipeline, error = load_pipeline()

if error:
    st.error(f"Error loading model: {error}")
    st.info("Make sure you've saved your model and preprocessor using pickle:")
    st.code("""
    # Save your model and preprocessor
    with open('model.pkl', 'wb') as file:
        pickle.dump(final_pipeline.named_steps['classifier'], file)
        
    with open('preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)
    """)
    st.stop()

# Sidebar for information
st.sidebar.header("About")
st.sidebar.info(
    """
    This app predicts whether a client will subscribe to a term deposit 
    based on demographic and banking information.
    
    The model was trained on the Bank Marketing dataset.
    """
)

# Create two columns for better layout
col1, col2 = st.columns(2)

# Input form
with st.form("prediction_form"):
    st.markdown('<p class="subheader">Client Information</p>', unsafe_allow_html=True)
    
    # Personal information
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=95, value=40)
        job_type = st.selectbox(
            "Job Type", 
            options=["admin.", "blue-collar", "entrepreneur", "housemaid", 
                    "management", "retired", "self-employed", "services", 
                    "student", "technician", "unemployed", "unknown"]
        )
        marital_status = st.selectbox(
            "Marital Status",
            options=["married", "single", "divorced"]
        )
        education = st.selectbox(
            "Education Level",
            options=["primary", "secondary", "tertiary", "unknown"]
        )
        credit_default_flag = st.selectbox(
            "Has Credit in Default?",
            options=["no", "yes"]
        )
    
    with col2:
        housing_loan_flag = st.selectbox(
            "Has Housing Loan?",
            options=["no", "yes"]
        )
        personal_loan_flag = st.selectbox(
            "Has Personal Loan?",
            options=["no", "yes"]
        )
        avg_yearly_balance = st.number_input("Average Yearly Balance (€)", min_value=-10000, max_value=100000, value=0)
        contact_type = st.selectbox(
            "Contact Communication Type",
            options=["cellular", "telephone", "unknown"]
        )
        last_contact_day = st.slider("Day of Month Contacted", min_value=1, max_value=31, value=15)
    
    # Campaign information
    st.markdown('<p class="subheader">Campaign Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        last_contact_month = st.selectbox(
            "Month Contacted",
            options=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        )
        previous_campaign_outcome = st.selectbox(
            "Previous Campaign Outcome",
            options=["failure", "success", "nonexistent"]
        )
        campaign_contact_count = st.slider("Number of Contacts During Campaign", min_value=1, max_value=50, value=2)
    
    with col2:
        days_since_prev_contact = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=-1, 
                              help="-1 means client was not previously contacted")
        previous_contacts_count = st.slider("Previous Contacts Before Campaign", min_value=0, max_value=20, value=0)
        
    # Submit button
    submitted = st.form_submit_button("Predict Subscription", type="primary")

# Add this before loading your model - define the exact feature order
def get_original_columns():
    # Define the numeric and categorical features - NO economic indicators
    numeric_features = [
        'age', 'avg_yearly_balance', 'last_contact_day',
        'campaign_contact_count', 'days_since_prev_contact', 'previous_contacts_count'
    ]
    
    categorical_features = [
        'job_type', 'marital_status', 'education',
        'credit_default_flag', 'housing_loan_flag',
        'personal_loan_flag', 'contact_type',
        'last_contact_month', 'previous_campaign_outcome'
    ]
    
    return numeric_features, categorical_features

# Make prediction when the form is submitted
if submitted:
    try:
        # Create input data DataFrame with ONLY the 15 expected features
        input_data = pd.DataFrame({
            'age': [age],
            'avg_yearly_balance': [avg_yearly_balance],
            'last_contact_day': [last_contact_day],
            'campaign_contact_count': [campaign_contact_count],
            'days_since_prev_contact': [days_since_prev_contact],
            'previous_contacts_count': [previous_contacts_count],
            'job_type': [job_type],
            'marital_status': [marital_status],
            'education': [education],
            'credit_default_flag': [credit_default_flag],
            'housing_loan_flag': [housing_loan_flag],
            'personal_loan_flag': [personal_loan_flag],
            'contact_type': [contact_type],
            'last_contact_month': [last_contact_month],
            'previous_campaign_outcome': [previous_campaign_outcome]
        })
        
        # No references to economic indicators anywhere in this code
        st.write(f"Input shape before processing: {input_data.shape}")
        
        # Now transform the data
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0, 1]
        
        # Display result
        st.markdown("---")
        st.markdown('<p class="subheader">Prediction Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 'Yes':
                st.success(f"## Client WILL subscribe to term deposit")
            else:
                st.warning(f"## Client WILL NOT subscribe to term deposit")
            
            st.write(f"Probability: {probability:.2%}")
        
        with col2:
            # Create a gauge chart for probability
            fig, ax = plt.subplots(figsize=(8, 3))
            
            # Create gauge
            gauge_colors = [(0.3, 0, 0, 0.6), (0.6, 0.3, 0, 0.8), (0, 0.6, 0.3, 1.0)]
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(0, 1)
            
            # Plot the gauge background
            ax.barh(0, 1, height=0.5, color='lightgray', alpha=0.3)
            
            # Plot the value as a bar
            ax.barh(0, probability, height=0.5, color=cmap(norm(probability)))
            
            # Add a marker for the threshold
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels
            ax.text(0, -0.25, "0%", ha='center', va='center')
            ax.text(0.5, -0.25, "50%", ha='center', va='center')
            ax.text(1, -0.25, "100%", ha='center', va='center')
            ax.text(0.5, 0.75, "Subscription Probability", ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Add the actual value
            ax.text(probability, 0, f"{probability:.1%}", ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # Clean up the axes
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_axis_off()
            
            st.pyplot(fig)
        
        # Feature importance explanation (simplified)
        st.markdown("### Most Important Features")
        st.info("""
        Banking clients are more likely to subscribe when:
        - They had a successful previous campaign outcome
        - They were contacted via cellular phone
        - They have a higher education level
        - They were contacted during specific months (March, September, October, December)
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Debug information:")
        st.write(f"Input data shape: {input_data.shape}")
        st.write("Input data preview:")
        st.write(input_data)
        
# Add explanatory information
with st.expander("💡 How to Use This App"):
    st.write("""
    1. Fill in the client's personal and banking information
    2. Add campaign details
    3. Click "Predict Subscription" to get the result
    4. The gauge shows the probability of subscription
    
    This model helps banks target clients who are more likely to subscribe to term deposits, 
    optimizing marketing campaigns and increasing success rates.
    """)
    
with st.expander("📊 About the Model"):
    st.write("""
    This prediction uses a Logistic Regression model trained on the Bank Marketing dataset.
    
    **Model Performance Metrics:**
    - Accuracy: ~90%
    - ROC-AUC: ~0.93
    - Precision: ~65%
    - Recall: ~50%
    
    The model uses various client attributes to predict 
    the likelihood of term deposit subscription.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 | Bank Term Deposit Prediction App")
