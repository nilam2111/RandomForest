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

# Custom CSS for layout, animations, and button styling
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid #1e293b;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Input field cards */
    .css-1r6slb0, .stSelectbox, .stNumberInput {
        background-color: #1e293b;
        border-radius: 8px;
    }
    
    /* Custom styled Predict Button */
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

    /* Success / Warning banner enhancements */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# Header Section
st.title("🔮 Customer Prediction Dashboard")
st.caption("Provide customer demographics below to generate target predictions using the loaded RandomForest model.")
st.markdown("---")

# Main Form Layout
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

# Prediction Logic and Visual Effects
if submit_btn:
    # Trigger visual effect: Snow/Balloons + Spinner
    st.snow()
    
    with st.spinner("Analyzing input patterns and running prediction..."):
        time.sleep(1.2)  # Short artificial delay to emphasize effect
        
        # Format input data matching the feature order in model.pkl:
        # ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'Customer Type']
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Marital Status': marital_status,
            'Occupation': occupation,
            'Monthly Income': monthly_income,
            'Educational Qualifications': education,
            'Family size': family_size,
            'Customer Type': customer_type
        }])
        
        try:
            prediction = model.predict(input_data)[0]
            
            # Predict probabilities if supported
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_data)[0]
                classes = model.classes_
            else:
                probs = None

            st.markdown("---")
            st.subheader("🎯 Result")

            if str(prediction).lower() in ["yes", "1", "true"]:
                st.balloons()
                st.success(f"**Prediction Outcome:** {prediction}")
            else:
                st.info(f"**Prediction Outcome:** {prediction}")

            if probs is not None:
                st.write("**Prediction Probabilities:**")
                prob_df = pd.DataFrame([probs], columns=classes)
                st.bar_chart(prob_df.T)

        except Exception as err:
            st.error(f"Error during prediction: {err}")
            st.info("Make sure categorical values are pre-encoded if your model expects numerical features directly.")
