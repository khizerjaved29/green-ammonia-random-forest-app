
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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
TEMP_RESULTS_FILE = Path("Temperature_Predictions.xlsx")
NH3_RESULTS_FILE = Path("NH3_Predictions.xlsx")

FEATURES = [
    "Time (hr)",
    "H2 Load (%)",
    "N2 Load (%)",
    "Feed Flow Ramp (%/hr)",
    "H2:N2 Ratio Ramp Time (hr)",
    "H2:N2 Molar Ratio",
    "Controller_Mode"
]

@st.cache_resource
def load_models():
    return joblib.load(MODEL_TEMP), joblib.load(MODEL_NH3)

try:
    temperature_model, nh3_model = load_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error("The trained model files could not be loaded.")
    st.code(str(e))

# ---------------- INPUTS ----------------
st.subheader("Operating Inputs")
c1, c2 = st.columns(2)

with c1:
    time_hr = st.number_input("Time (hr)", min_value=0.0, max_value=8.0, value=3.0, step=0.1)
    h2_load = st.number_input("H₂ Load (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
    n2_load = st.number_input("N₂ Load (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
    feed_ramp = st.selectbox("Feed Flow Ramp (%/hr)", [0.0, 20.0, 100.0], index=1)

with c2:
    ratio_ramp_time = st.number_input(
        "H₂:N₂ Ratio Ramp Time (hr)",
        min_value=0.0, max_value=0.5, value=0.0, step=0.05
    )
    h2_n2_ratio = st.number_input(
        "H₂:N₂ Molar Ratio",
        min_value=2.5, max_value=3.0, value=3.0, step=0.05
    )
    controller_label = st.selectbox(
        "Controller Mode",
        ["Original / Base Controller", "Additional Temperature Controller"],
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
        st.error("Models are not loaded.")
    else:
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

# ---------------- METRICS ----------------
st.divider()
st.header("Model Validation Metrics")

performance = pd.DataFrame({
    "Output": [
        "Temperature Bed 1",
        "Temperature Bed 2",
        "Temperature Bed 3",
        "NH3 Bed 1",
        "NH3 Bed 2"
    ],
    "R²": [0.9248, 0.9956, 0.9982, 0.9657, 0.9743],
    "MAE": [0.5176, 0.8357, 0.6202, 0.00034, 0.00236],
    "RMSE": [0.9706, 1.5797, 0.9631, 0.00057, 0.00356]
})
st.dataframe(performance, use_container_width=True, hide_index=True)

# ---------------- ACTUAL VS PREDICTED ----------------
st.divider()
st.header("Actual vs Predicted Validation Graphs")

def actual_vs_predicted_plot(actual, predicted, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted)
    minimum = min(float(actual.min()), float(predicted.min()))
    maximum = max(float(actual.max()), float(predicted.max()))
    ax.plot([minimum, maximum], [minimum, maximum], "--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

if TEMP_RESULTS_FILE.exists():
    temp_results = pd.read_excel(TEMP_RESULTS_FILE)

    st.subheader("Temperature Model")

    col1, col2 = st.columns(2)
    with col1:
        actual_vs_predicted_plot(
            temp_results["T Bed 1 (°C)"],
            temp_results["Predicted Bed 1"],
            "Bed 1: Actual vs Predicted Temperature",
            "Actual Temperature (°C)",
            "Predicted Temperature (°C)"
        )
    with col2:
        actual_vs_predicted_plot(
            temp_results["T Bed 2 (°C)"],
            temp_results["Predicted Bed 2"],
            "Bed 2: Actual vs Predicted Temperature",
            "Actual Temperature (°C)",
            "Predicted Temperature (°C)"
        )

    actual_vs_predicted_plot(
        temp_results["T Bed 3 (°C)"],
        temp_results["Predicted Bed 3"],
        "Bed 3: Actual vs Predicted Temperature",
        "Actual Temperature (°C)",
        "Predicted Temperature (°C)"
    )
else:
    st.warning(
        "Temperature_Predictions.xlsx is not in the GitHub repository, "
        "so temperature Actual vs Predicted graphs cannot be displayed yet."
    )

if NH3_RESULTS_FILE.exists():
    nh3_results = pd.read_excel(NH3_RESULTS_FILE)

    st.subheader("NH₃ Model")

    col3, col4 = st.columns(2)
    with col3:
        actual_vs_predicted_plot(
            nh3_results["NH3 Bed 1 Mole Fraction"],
            nh3_results["Predicted NH3 Bed 1"],
            "NH3 Bed 1: Actual vs Predicted",
            "Actual NH3 Mole Fraction",
            "Predicted NH3 Mole Fraction"
        )
    with col4:
        actual_vs_predicted_plot(
            nh3_results["NH3 Bed 2 Mole Fraction"],
            nh3_results["Predicted NH3 Bed 2"],
            "NH3 Bed 2: Actual vs Predicted",
            "Actual NH3 Mole Fraction",
            "Predicted NH3 Mole Fraction"
        )
else:
    st.warning(
        "NH3_Predictions.xlsx is not in the GitHub repository, "
        "so NH₃ Actual vs Predicted graphs cannot be displayed yet."
    )

# ---------------- FEATURE IMPORTANCE ----------------
if models_ready:
    st.divider()
    st.header("Feature Importance")

    temp_importance = pd.DataFrame({
        "Input Variable": FEATURES,
        "Importance": temperature_model.feature_importances_
    }).sort_values("Importance", ascending=False).set_index("Input Variable")

    st.markdown("**Temperature Model Feature Importance**")
    st.bar_chart(temp_importance)

    nh3_importance = pd.DataFrame({
        "Input Variable": FEATURES,
        "Importance": nh3_model.feature_importances_
    }).sort_values("Importance", ascending=False).set_index("Input Variable")

    st.markdown("**NH₃ Model Feature Importance**")
    st.bar_chart(nh3_importance)

st.divider()
st.caption(
    "Model 1 predicts reactor-bed temperatures. "
    "Model 2 predicts NH₃ mole fractions for Beds 1 and 2."
)
