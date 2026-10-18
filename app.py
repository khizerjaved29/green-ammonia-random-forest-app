
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Green Ammonia Production Prediction System",
    page_icon="🧪",
    layout="wide"
)

st.title("Green Ammonia Production Prediction System")
st.caption(
    "Random Forest models for reactor-bed temperature and NH₃ mole-fraction prediction"
)

# ---------------- FILE PATHS ----------------
TEMP_MODEL_FILE = Path("Temperature_Random_Forest.pkl")
NH3_MODEL_FILE = Path("NH3_Random_Forest.pkl")
TEMP_RESULTS_FILE = Path("Temperature_Predictions.xlsx")
NH3_RESULTS_FILE = Path("NH3_Predictions.xlsx")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    temperature_model = joblib.load(TEMP_MODEL_FILE)
    nh3_model = joblib.load(NH3_MODEL_FILE)
    return temperature_model, nh3_model

try:
    temperature_model, nh3_model = load_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error("Model files could not be loaded.")
    st.code(str(e))

# =========================================================
# 1. OPERATING INPUTS
# =========================================================
st.header("Operating Inputs")

col1, col2 = st.columns(2)

with col1:
    time_hr = st.number_input(
        "Time (hr)",
        min_value=0.0,
        max_value=8.0,
        value=3.0,
        step=0.1
    )

    h2_load = st.number_input(
        "H₂ Load (%)",
        min_value=10.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    n2_load = st.number_input(
        "N₂ Load (%)",
        min_value=10.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    feed_ramp = st.selectbox(
        "Feed Flow Ramp (%/hr)",
        options=[0.0, 20.0, 100.0],
        index=1
    )

with col2:
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
        options=[
            "Original / Base Controller",
            "Additional Temperature Controller"
        ],
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

# =========================================================
# 2. LIVE MODEL PREDICTIONS
# =========================================================
if st.button("Predict Outputs", type="primary", use_container_width=True):

    if not models_ready:
        st.error("Models are not loaded. Check both .pkl files in GitHub.")
    else:
        try:
            temp_output = temperature_model.predict(new_input)
            nh3_output = nh3_model.predict(new_input)

            st.header("Predicted Outputs")

            a, b, c = st.columns(3)

            a.metric(
                "Bed 1 Temperature",
                f"{temp_output[0][0]:.2f} °C"
            )

            b.metric(
                "Bed 2 Temperature",
                f"{temp_output[0][1]:.2f} °C"
            )

            c.metric(
                "Bed 3 Temperature",
                f"{temp_output[0][2]:.2f} °C"
            )

            d, e = st.columns(2)

            d.metric(
                "NH₃ Bed 1 Mole Fraction",
                f"{nh3_output[0][0]:.5f}"
            )

            e.metric(
                "NH₃ Bed 2 Mole Fraction",
                f"{nh3_output[0][1]:.5f}"
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

# =========================================================
# 3. MODEL VALIDATION METRICS
# =========================================================
st.divider()
st.header("Model Validation Metrics")

metrics_df = pd.DataFrame({
    "Output": [
        "Temperature Bed 1",
        "Temperature Bed 2",
        "Temperature Bed 3",
        "NH₃ Bed 1 Mole Fraction",
        "NH₃ Bed 2 Mole Fraction"
    ],
    "R²": [
        0.9248,
        0.9956,
        0.9982,
        0.9657,
        0.9743
    ],
    "MAE": [
        0.5176,
        0.8357,
        0.6202,
        0.00034,
        0.00236
    ],
    "RMSE": [
        0.9706,
        1.5797,
        0.9631,
        0.00057,
        0.00356
    ]
})

st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 4. ACTUAL VS PREDICTED GRAPHS
# =========================================================
st.divider()
st.header("Actual vs Predicted Validation Graphs")

st.write(
    "The dashed diagonal line represents ideal agreement "
    "(Actual = Predicted). Points closer to this line indicate better prediction."
)

def make_actual_vs_predicted_plot(
    actual,
    predicted,
    title,
    xlabel,
    ylabel
):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Actual vs predicted points
    ax.scatter(actual, predicted, s=55)

    # Ideal line: Actual = Predicted
    minimum = min(float(actual.min()), float(predicted.min()))
    maximum = max(float(actual.max()), float(predicted.max()))

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--",
        linewidth=2
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig

# ---------- TEMPERATURE GRAPHS ----------
if TEMP_RESULTS_FILE.exists():

    try:
        temp_results = pd.read_excel(TEMP_RESULTS_FILE)

        st.subheader("Temperature Model")

        t1, t2 = st.columns(2)

        with t1:
            fig1 = make_actual_vs_predicted_plot(
                temp_results["T Bed 1 (°C)"],
                temp_results["Predicted Bed 1"],
                "Bed 1: Actual vs Predicted Temperature",
                "Actual Temperature (°C)",
                "Predicted Temperature (°C)"
            )
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with t2:
            fig2 = make_actual_vs_predicted_plot(
                temp_results["T Bed 2 (°C)"],
                temp_results["Predicted Bed 2"],
                "Bed 2: Actual vs Predicted Temperature",
                "Actual Temperature (°C)",
                "Predicted Temperature (°C)"
            )
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        fig3 = make_actual_vs_predicted_plot(
            temp_results["T Bed 3 (°C)"],
            temp_results["Predicted Bed 3"],
            "Bed 3: Actual vs Predicted Temperature",
            "Actual Temperature (°C)",
            "Predicted Temperature (°C)"
        )
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    except Exception as e:
        st.error("Temperature validation graphs could not be created.")
        st.code(str(e))

else:
    st.warning(
        "Temperature_Predictions.xlsx is missing from the GitHub repository."
    )

# ---------- NH3 GRAPHS ----------
if NH3_RESULTS_FILE.exists():

    try:
        nh3_results = pd.read_excel(NH3_RESULTS_FILE)

        st.subheader("NH₃ Model")

        n1, n2 = st.columns(2)

        with n1:
            fig4 = make_actual_vs_predicted_plot(
                nh3_results["NH3 Bed 1 Mole Fraction"],
                nh3_results["Predicted NH3 Bed 1"],
                "NH₃ Bed 1: Actual vs Predicted",
                "Actual NH₃ Mole Fraction",
                "Predicted NH₃ Mole Fraction"
            )
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        with n2:
            fig5 = make_actual_vs_predicted_plot(
                nh3_results["NH3 Bed 2 Mole Fraction"],
                nh3_results["Predicted NH3 Bed 2"],
                "NH₃ Bed 2: Actual vs Predicted",
                "Actual NH₃ Mole Fraction",
                "Predicted NH₃ Mole Fraction"
            )
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

    except Exception as e:
        st.error("NH₃ validation graphs could not be created.")
        st.code(str(e))

else:
    st.warning(
        "NH3_Predictions.xlsx is missing from the GitHub repository."
    )

st.divider()
st.caption(
    "Temperature model: Bed 1, Bed 2 and Bed 3 temperatures. "
    "NH₃ model: Bed 1 and Bed 2 mole fractions."
)
