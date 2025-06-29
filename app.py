import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("house_price_model.pkl")
features = joblib.load("features.pkl")

st.title("House Price Prediction App")
st.markdown("Enter property details to estimate the sale price.")

# Input widgets
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Grade Living Area (sqft)", 500, 5000, 1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 800)
full_bath = st.slider("Number of Full Bathrooms", 0, 4, 2)

# Make prediction
if st.button("Predict Sale Price"):
    input_df = pd.DataFrame([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, full_bath]], columns=features)
    prediction = model.predict(input_df)[0]
    st.success(f" Estimated Sale Price: ₹ {int(prediction):,}")
