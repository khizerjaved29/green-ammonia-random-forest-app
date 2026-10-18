
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Green Ammonia Production Prediction System",
    page_icon="🧪",
    layout="wide"
)

st.title("Green Ammonia Production Prediction System")
st.caption("Random Forest models for reactor-bed temperature and NH₃ mole-fraction prediction")

MODEL_TEMP = Path("Temperature_Random_Forest.pkl")
MODEL_NH3 = Path("NH3_Random_Forest.pkl")

@st.cache_resource
def load_models():
    if not MODEL_TEMP.exists() or not MODEL_NH3.exists():
        missing = []
        if not MODEL_TEMP.exists():
            missing.append(str(MODEL_TEMP))
        if not MODEL_NH3.exists():
            missing.append(str(MODEL_NH3))
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

    temperature_model = joblib.load(MODEL_TEMP)
    nh3_model = joblib.load(MODEL_NH3)
    return temperature_model, nh3_model

try:
    temperature_model, nh3_model = load_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(
        "Model files could not be loaded. Make sure these files are in the same GitHub repository:\n\n"
        "- Temperature_Random_Forest.pkl\n"
        "- NH3_Random_Forest.pkl"
    )
    st.code(str(e))

st.subheader("Operating Inputs")

c1, c2 = st.columns(2)

with c1:
    time_hr = st.number_input("Time (hr)", min_value=0.0, max_value=8.0, value=3.0, step=0.1)
    h2_load = st.number_input("H₂ Load (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
    n2_load = st.number_input("N₂ Load (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
    feed_ramp = st.selectbox("Feed Flow Ramp (%/hr)", options=[0.0, 20.0, 100.0], index=1)

with c2:
    ratio_ramp_time = st.number_input(
        "H₂:N₂ Ratio Ramp Time (hr)",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.05
    )
    h2_n2_ratio = st.number_input(
        "H₂:N₂ Molar Ratio",
        min_value=2.5,
        max_value=3.0,
        value=3.0,
        step=0.05
    )
    controller_label = st.selectbox(
        "Controller Mode",
        options=["Original / Base Controller", "Additional Temperature Controller"],
        index=1
    )
    controller_mode = 0 if controller_label == "Original / Base Controller" else 1

st.info(
    "Use inputs mainly within the ranges represented by the training data. "
    "Predictions outside the training range may be unreliable."
)

new_input = pd.DataFrame({
    "Time (hr)": [time_hr],
    "H2 Load (%)": [h2_load],
    "N2 Load (%)": [n2_load],
    "Feed Flow Ramp (%/hr)": [feed_ramp],
    "H2:N2 Ratio Ramp Time (hr)": [ratio_ramp_time],
    "H2:N2 Molar Ratio": [h2_n2_ratio],
    "Controller_Mode": [controller_mode]
})

st.subheader("Selected Input Conditions")
st.dataframe(new_input, use_container_width=True, hide_index=True)

if st.button("Predict Outputs", type="primary", use_container_width=True):
    if not models_ready:
        st.error("Models are not loaded yet. Upload both .pkl files to the repository first.")
    else:
        try:
            temp_output = temperature_model.predict(new_input)
            nh3_output = nh3_model.predict(new_input)

            st.subheader("Predicted Outputs")

            a, b, c = st.columns(3)
            a.metric("Bed 1 Temperature", f"{temp_output[0][0]:.2f} °C")
            b.metric("Bed 2 Temperature", f"{temp_output[0][1]:.2f} °C")
            c.metric("Bed 3 Temperature", f"{temp_output[0][2]:.2f} °C")

            d, e = st.columns(2)
            d.metric("NH₃ Bed 1 Mole Fraction", f"{nh3_output[0][0]:.5f}")
            e.metric("NH₃ Bed 2 Mole Fraction", f"{nh3_output[0][1]:.5f}")

            results = pd.DataFrame({
                "Output": [
                    "Bed 1 Temperature (°C)",
                    "Bed 2 Temperature (°C)",
                    "Bed 3 Temperature (°C)",
                    "NH3 Bed 1 Mole Fraction",
                    "NH3 Bed 2 Mole Fraction"
                ],
                "Predicted Value": [
                    float(temp_output[0][0]),
                    float(temp_output[0][1]),
                    float(temp_output[0][2]),
                    float(nh3_output[0][0]),
                    float(nh3_output[0][1])
                ]
            })

            st.subheader("Results Table")
            st.dataframe(results, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

st.divider()
st.caption(
    "Model 1 predicts reactor-bed temperatures. "
    "Model 2 predicts NH₃ mole fractions for Beds 1 and 2."
)
