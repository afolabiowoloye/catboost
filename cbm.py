import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd

# Load the trained model
model = CatBoostRegressor()
model.load_model('catboost_regression_model.cbm')

# Streamlit UI
st.title("CatBoost Regression Predictor")

# Input fields (modify based on your features)
feature1 = st.number_input("Feature 1", value=0.5)
feature2 = st.number_input("Feature 2", value=0.5)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[feature1, feature2]], columns=['feature1', 'feature2'])
    prediction = model.predict(input_data)
    st.success(f"Predicted Target: {prediction[0]:.2f}")
