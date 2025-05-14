# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:39:06 2025

@author: DELL
"""
import pandas as pd
import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgboost_model9.joblib")
scaler = joblib.load("minmax_scaler9.joblib")

# Streamlit App
st.title("Digital Marketing Campaign: Conversion Prediction")

# User Inputs
EmailClicks = st.number_input("Email Clicks", min_value=1, step=1)
TimeOnSite = st.number_input("Time Spent on Site (minutes)", min_value=0, step=1)
AdSpend = st.number_input("Ad Spend (in $)", min_value=0, step=10)
ClickThroughRate = st.number_input("Click Through Rate (CTR)", min_value=0.0, step=0.01)
EmailOpens = st.number_input("Email Opens", min_value=0, step=1)

# Predict Button
if st.button("Predict Conversion"):
    # Prepare Input
    input_data = np.array([[EmailClicks, TimeOnSite, AdSpend, ClickThroughRate, EmailOpens]])
    input_scaled = scaler.transform(input_data)
    
    # Make Prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display Result
    if prediction == 1:
        st.success("✅ This customer is likely to convert!")
    else:
        st.error("❌ This customer is unlikely to convert.")