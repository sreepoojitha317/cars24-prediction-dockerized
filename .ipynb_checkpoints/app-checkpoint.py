import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessor
@st.cache_resource
def load_assets():
    # Make sure these filenames match what you saved from your notebook
    model = joblib.load('xgb_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_assets()

st.title("🚗 Used Car Price Predictor (Updated)")
st.info("Please fill in all location and registration details to avoid missing column errors.")

# Create Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Car Basics")
    make = st.selectbox("Make", ['Maruti', 'Hyundai', 'Honda', 'Skoda', 'Volkswagen'])
    model_name = st.text_input("Model", "Swift")
    year = st.number_input("Year", 2000, 2024, 2018)
    kilometerdriven = st.number_input("Kms Driven", 0, 500000, 40000)

with col2:
    st.subheader("Technical")
    fueltype = st.selectbox("Fuel", ['Petrol', 'Diesel', 'Petrol + Cng'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    bodytype = st.selectbox("Body", ['Hatchback', 'Sedan', 'SUV', 'Unknown'])
    ownernumber = st.slider("Owners", 1, 5, 1)

with col3:
    st.subheader("Location & Reg.")
    city = st.text_input("Current City", "New Delhi")
    storename = st.text_input("Store Name", "In-Store")
    reg_city = st.text_input("Registration City", "Delhi")
    reg_state = st.text_input("Registration State", "Delhi")

# Sidebar for hidden/numeric features
st.sidebar.header("Additional Pricing Factors")
benefits = st.sidebar.number_input("Benefits", value=0)
discountprice = st.sidebar.number_input("Discount Price", value=0)
isc24assured = st.sidebar.checkbox("C24 Assured", value=True)

# Calculation for car_age
car_age = 2024 - year

# Assemble the DataFrame with ALL missing columns
input_dict = {
    'make': make,
    'model': model_name,
    'year': year,
    'fueltype': fueltype,
    'kilometerdriven': kilometerdriven,
    'ownernumber': ownernumber,
    'transmission': transmission,
    'bodytype': bodytype,
    'registrationcity': reg_city,    # Added
    'registrationstate': reg_state,  # Added
    'city': city,                    # Added
    'storename': storename,          # Added
    'isc24assured': isc24assured,
    'benefits': benefits,
    'discountprice': discountprice,
    'car_age': car_age
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Price", type="primary"):
    try:
        # Preprocess and Predict
        processed_data = preprocessor.transform(input_df)
        prediction = model.predict(processed_data)

        st.success(f"### Estimated Market Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Current Columns in Input:", input_df.columns.tolist())
