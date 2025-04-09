import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd
import requests
import io

# Title
st.title("CatBoost Regression Predictor")

# Load the model from GitHub (raw URL)
MODEL_URL = "https://github.com/afolabiowoloye/catboost/raw/refs/heads/main/model/catboost_regression_model.cbm"

@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()  # Check for download errors
    model = CatBoostRegressor()
    model.load_model(io.BytesIO(response.content))
    return model

model = load_model()

# Input fields (modify based on your features)
feature1 = st.number_input("Feature 1", value=0.5)
feature2 = st.number_input("Feature 2", value=0.5)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[feature1, feature2]], columns=['feature1', 'feature2'])
    prediction = model.predict(input_data)
    st.success(f"Predicted Target: {prediction[0]:.2f}")
