import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🔥 Algerian Forest Fire Risk Prediction App")
st.write("Enter weather data to predict Fire Weather Index (FWI)")

temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)
RH = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0)
Ws = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0)
Rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0)
FFMC = st.number_input("FFMC Index", min_value=0.0, max_value=150.0)
DMC = st.number_input("DMC Index", min_value=0.0, max_value=200.0)
DC = st.number_input("DC Index", min_value=0.0, max_value=1000.0)
ISI = st.number_input("ISI Index", min_value=0.0, max_value=50.0)
BUI = st.number_input("BUI Index", min_value=0.0, max_value=200.0)

if st.button("Predict FWI"):
    input_data = np.array([[temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"🔥 Predicted FWI: {prediction:.2f}")
