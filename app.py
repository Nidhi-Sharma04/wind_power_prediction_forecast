import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Wind Power Predictor", page_icon="🍃", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("wind_power_model.sav")

model = load_model()

FEATURES = [
    'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
    'winddirection_100m', 'windgusts_10m',
    'Location_Location1.csv', 'Location_Location2.csv',
    'Location_Location3.csv', 'Location_Location4.csv'
]

st.title("🍃 Wind Power Generation Predictor")
st.markdown("Predict power output using a **Random Forest** model — R² = **0.6752**")
st.divider()

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Input Parameters")

location = st.sidebar.selectbox("📍 Location", ["Location1", "Location2", "Location3", "Location4"])

st.sidebar.subheader("🌡️ Atmospheric Conditions")
temperature = st.sidebar.slider("Temperature (2m) °C",    0.0, 45.0, 28.5, 0.1)
humidity    = st.sidebar.slider("Relative Humidity (%)",    0,  100,   85)
dewpoint    = st.sidebar.slider("Dew Point (2m) °C",      0.0, 40.0, 24.5, 0.1)

st.sidebar.subheader("💨 Wind Conditions")
windspeed_10  = st.sidebar.slider("Wind Speed 10m (km/h)",   0.0, 50.0,  5.0, 0.1)
windspeed_100 = st.sidebar.slider("Wind Speed 100m (km/h)",  0.0, 80.0,  8.0, 0.1)
winddir_10    = st.sidebar.slider("Wind Direction 10m (°)",    0,  360,  180)
winddir_100   = st.sidebar.slider("Wind Direction 100m (°)",   0,  360,  175)
windgusts     = st.sidebar.slider("Wind Gusts 10m (km/h)",   0.0, 80.0, 10.0, 0.1)

# ── Build input row ────────────────────────────────────────────────────────────
input_dict = {
    'temperature_2m':       temperature,
    'relativehumidity_2m':  humidity,
    'dewpoint_2m':          dewpoint,
    'windspeed_10m':        windspeed_10,
    'windspeed_100m':       windspeed_100,
    'winddirection_10m':    winddir_10,
    'winddirection_100m':   winddir_100,
    'windgusts_10m':        windgusts,
    'Location_Location1.csv': 1 if location == "Location1" else 0,
    'Location_Location2.csv': 1 if location == "Location2" else 0,
    'Location_Location3.csv': 1 if location == "Location3" else 0,
    'Location_Location4.csv': 1 if location == "Location4" else 0,
}

input_df = pd.DataFrame([input_dict])[FEATURES]
prediction = float(np.clip(model.predict(input_df)[0], 0, 1))

# ── Output ─────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("⚡ Predicted Power", f"{prediction:.4f}")
col2.metric("📊 Capacity Used",   f"{prediction*100:.1f}%")

if prediction < 0.2:
    level = "🔴 Low"
elif prediction < 0.5:
    level = "🟡 Medium"
else:
    level = "🟢 High"
col3.metric("🏷️ Output Level", level)

st.divider()
st.subheader("⚡ Power Output Gauge")
st.progress(prediction)
st.caption(f"Power output is **{prediction*100:.1f}%** of maximum capacity")

# ── Input summary ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Input Summary")
c1, c2, c3 = st.columns(3)
params = [
    ("📍 Location", location),
    ("🌡️ Temperature", f"{temperature} °C"),
    ("💧 Humidity", f"{humidity}%"),
    ("🌫️ Dew Point", f"{dewpoint} °C"),
    ("💨 Wind Speed 10m", f"{windspeed_10} km/h"),
    ("💨 Wind Speed 100m", f"{windspeed_100} km/h"),
    ("🧭 Wind Dir 10m", f"{winddir_10}°"),
    ("🧭 Wind Dir 100m", f"{winddir_100}°"),
    ("🌪️ Wind Gusts", f"{windgusts} km/h"),
]
for i, col in enumerate([c1, c2, c3]):
    for k, v in params[i*3:(i+1)*3]:
        col.metric(k, v)

# ── Batch prediction ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📂 Batch Prediction (Upload CSV)")
uploaded = st.file_uploader("Upload CSV with same columns as training data", type=["csv"])
if uploaded:
    try:
        batch = pd.read_csv(uploaded)
        batch = batch.drop(columns=['Time'], errors='ignore')
        batch = pd.get_dummies(batch, columns=['Location'], drop_first=False)
        for col in FEATURES:
            if col not in batch.columns:
                batch[col] = 0
        batch['Predicted_Power'] = np.clip(model.predict(batch[FEATURES]), 0, 1)
        st.success(f"✅ Predictions generated for {len(batch)} rows!")
        st.dataframe(batch[['windspeed_100m', 'windgusts_10m', 'Predicted_Power']].head(50))
        st.download_button("⬇️ Download Predictions CSV",
                           batch.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Model: Random Forest | R² = 0.6752 | 175,200 records | 4 locations")
