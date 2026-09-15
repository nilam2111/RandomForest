import time
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Customer Insight Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(168, 85, 247, 0.39);
        transition: all 0.3s ease-in-out;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(168, 85, 247, 0.6);
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    }
    .result-card {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 15px;
    }
    .result-yes {
        background-color: #064e3b;
        color: #34d399;
        border: 2px solid #10b981;
    }
    .result-no {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 2px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_assets():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Header Section
st.title("🔮 Customer Prediction Dashboard")
st.caption("Provide customer demographics below to generate target predictions using the loaded model.")
st.markdown("---")

# Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("📋 Demographic Profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", options=["Student", "Employee", "Self Employed", "Unemployed", "Housewife"])

    with col2:
        st.subheader("📊 Socioeconomic Details")
        monthly_income = st.selectbox(
            "Monthly Income", 
            options=["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"]
        )
        education = st.selectbox(
            "Educational Qualifications", 
            options=["Under Graduate", "Post Graduate", "School", "Ph.D", "Uneducated"]
        )
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3, step=1)
        customer_type = st.selectbox("Customer Type", options=["Direct", "Indirect", "Walk-in", "Online"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("🚀 Generate Prediction")

# Prediction Logic
if submit_btn:
    st.snow()
    
    with st.spinner("Analyzing input patterns and running prediction..."):
        time.sleep(1)
        
        # Encoding mapping
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
        occupation_map = {"Student": 0, "Employee": 1, "Self Employed": 2, "Unemployed": 3, "Housewife": 4}
        income_map = {"No Income": 0, "Below Rs.10000": 1, "10001 to 25000": 2, "25001 to 50000": 3, "More than 50000": 4}
        education_map = {"Under Graduate": 0, "Post Graduate": 1, "School": 2, "Ph.D": 3, "Uneducated": 4}
        customer_type_map = {"Direct": 0, "Indirect": 1, "Walk-in": 2, "Online": 3}

        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender_map[gender],
            'Marital Status': marital_map[marital_status],
            'Occupation': occupation_map[occupation],
            'Monthly Income': income_map[monthly_income],
            'Educational Qualifications': education_map[education],
            'Family size': family_size,
            'Customer Type': customer_type_map[customer_type]
        }])
        
        try:
            raw_prediction = model.predict(input_data)[0]
            
            # Map output strictly to YES or NO
            if str(raw_prediction).lower() in ["1", "yes", "true"]:
                final_output = "YES"
            else:
                final_output = "NO"

            st.markdown("---")
            st.subheader("🎯 Result")

            # Display styled Yes/No Output
            if final_output == "YES":
                st.balloons()
                st.markdown('<div class="result-card result-yes">Prediction Output: YES</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card result-no">Prediction Output: NO</div>', unsafe_allow_html=True)

        except Exception as err:
            st.error(f"Error during prediction: {err}")
