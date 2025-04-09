import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
np.__version__ = '1.21.6'
import requests
import io

# Title
st.title("CatBoost Regression Predictor")


@st.cache_resource(show_spinner=False)
def load_model():
    import os
    import urllib.request
    
    MODEL_URL = "https://github.com/afolabiowoloye/catboost/raw/refs/heads/main/model/catboost_regression_model.cbm"
    MODEL_PATH = "model.cbm"
    
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = CatBoostRegressor()
    return model.load_model(MODEL_PATH)

model = load_model()


# Load the model from GitHub (raw URL)
#MODEL_URL = "https://github.com/afolabiowoloye/catboost/raw/refs/heads/main/model/catboost_regression_model.cbm"


##@st.cache_resource  # Cache the model to avoid reloading on every interaction
#@st.cache_resource(ttl=3600)  # Refresh cache hourly
#def load_model():
#    response = requests.get(MODEL_URL)
 #   response.raise_for_status()  # Check for download errors
  #  model = CatBoostRegressor()
   # model.load_model(io.BytesIO(response.content))
    #return model
#model = load_model()



# Input fields (modify based on your features)
feature1 = st.number_input("Feature 1", value=0.5)
feature2 = st.number_input("Feature 2", value=0.5)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[feature1, feature2]], columns=['feature1', 'feature2'])
    prediction = model.predict(input_data)
    st.success(f"Predicted Target: {prediction[0]:.2f}")
