import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64

# Page config
st.set_page_config(
    page_title=" Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stNumberInput input {
        border-radius: 5px;
    }
    .metric-box {
        padding: 15px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
model_data = joblib.load('random_forest_model.pkl')
loaded_encoder = joblib.load('label_encoders.pkl')
loaded_model = model_data['model']

# Title
st.title(" Predictive Maintenance Model")
st.markdown("Predict **Mean Time to Failure (MTTF)** for Industrial Equipment")

# Input layout
st.markdown("### 📝 Enter Equipment Details")
col1, col2, col3 = st.columns(3)

with col1:
    product_type = st.selectbox("Product Type", ['Gauge Machine', 'Extruder', 'Pump', 'Coil Oven', 'Pressure Cutter'])

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

with col3:
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=150.0, step=0.1)

col4, col5 = st.columns(2)

with col4:
    age = st.number_input("Equipment Age (years)", min_value=0, max_value=100, step=1)

with col5:
    qty = st.number_input("Quantity", min_value=1, step=1)

# Predict button
st.markdown("### 📊 Prediction Result")
if st.button("🚀 Predict MTTF"):
    try:
        input_df = pd.DataFrame({
            'ProductType': [product_type],
            'Humidity': [humidity],
            'Temperature': [temperature],
            'Age': [age],
            'Quantity': [qty]
        })

        # Label Encoding
        for col, le in loaded_encoder.items():
            input_df[col] = le.transform(input_df[col])

        # Prediction
        prediction = loaded_model.predict(input_df)[0]

        # Output box
        st.markdown(f"""
            <div class="metric-box">
                <h2>🕒 Predicted MTTF</h2>
                <h1 style='color: #4CAF50;'>{prediction:.2f} Hours</h1>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
