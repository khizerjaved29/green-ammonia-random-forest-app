# Green Ammonia Random Forest App

This Streamlit application uses two trained Random Forest regression models:

1. `Temperature_Random_Forest.pkl`
   - Bed 1 Temperature
   - Bed 2 Temperature
   - Bed 3 Temperature

2. `NH3_Random_Forest.pkl`
   - NH3 Bed 1 Mole Fraction
   - NH3 Bed 2 Mole Fraction

## Required GitHub files

Upload these files to the root of the repository:

- `app.py`
- `requirements.txt`
- `Temperature_Random_Forest.pkl`
- `NH3_Random_Forest.pkl`

## Inputs

- Time (hr)
- H2 Load (%)
- N2 Load (%)
- Feed Flow Ramp (%/hr)
- H2:N2 Ratio Ramp Time (hr)
- H2:N2 Molar Ratio
- Controller Mode

## Deploy on Streamlit Community Cloud

Choose this GitHub repository and set the main file path to:

`app.py`

The app will automatically load both `.pkl` model files from the repository.
