import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Simulated PV system dataset
data = {
    'temperature_C': [25, 45, 35, 20, 50, 30, 40, 15, 33, 42],
    'voltage_V': [34, 28, 32, 36, 22, 33, 30, 37, 29, 25],
    'current_A': [5.5, 4.0, 5.0, 5.8, 3.5, 5.2, 4.8, 6.0, 4.2, 3.8],
    'power_W': [187, 112, 160, 209, 77, 172, 144, 222, 122, 95],
    'fault': [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
y = df['fault']
X = df.drop('fault', axis=1)
model = DecisionTreeClassifier()
model.fit(X, y)

st.title("AI PV Fault Detection Web App")
st.write("Enter PV system parameters to predict fault:")

temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0)
voltage = st.number_input("Voltage (V)", min_value=0.0, max_value=50.0, value=30.0)
current = st.number_input("Current (A)", min_value=0.0, max_value=10.0, value=5.0)
power = st.number_input("Power (W)", min_value=0.0, max_value=300.0, value=150.0)

if st.button("Predict Fault"):
    sample = [[temperature, voltage, current, power]]
    prediction = model.predict(sample)[0]
    if prediction == 1:
        st.error("Fault Detected!")
    else:
        st.success("No Fault Detected.")
