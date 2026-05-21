import streamlit as st

from src.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)


st.title("🚗 Used Car Price Predictor")

st.write("Enter Car Details")


col1, col2, col3 = st.columns(3)


with col1:

    make = st.selectbox(
        "Make",
        ['Maruti', 'Hyundai', 'Honda']
    )

    model = st.text_input(
        "Model",
        "Swift"
    )

    year = st.number_input(
        "Year",
        2000,
        2024,
        2018
    )

    kilometerdriven = st.number_input(
        "Kilometers Driven",
        0,
        500000,
        40000
    )


with col2:

    fueltype = st.selectbox(
        "Fuel Type",
        ['Petrol', 'Diesel']
    )

    transmission = st.selectbox(
        "Transmission",
        ['Manual', 'Automatic']
    )

    bodytype = st.selectbox(
        "Body Type",
        ['Hatchback', 'Sedan', 'SUV']
    )

    ownernumber = st.slider(
        "Owner Number",
        1,
        5,
        1
    )


with col3:

    city = st.text_input(
        "City",
        "Delhi"
    )

    storename = st.text_input(
        "Store Name",
        "In Store"
    )

    registrationcity = st.text_input(
        "Registration City",
        "Delhi"
    )

    registrationstate = st.text_input(
        "Registration State",
        "Delhi"
    )


benefits = st.sidebar.number_input(
    "Benefits",
    value=0
)

discountprice = st.sidebar.number_input(
    "Discount Price",
    value=0
)

isc24assured = st.sidebar.checkbox(
    "C24 Assured",
    value=True
)


car_age = 2024 - year


if st.button("Predict Price"):

    try:

        data = CustomData(

            make=make,

            model=model,

            year=year,

            fueltype=fueltype,

            kilometerdriven=kilometerdriven,

            ownernumber=ownernumber,

            transmission=transmission,

            bodytype=bodytype,

            registrationcity=registrationcity,

            registrationstate=registrationstate,

            city=city,

            storename=storename,

            isc24assured=isc24assured,

            benefits=benefits,

            discountprice=discountprice,

            car_age=car_age

        )


        pred_df = data.get_data_as_dataframe()


        predict_pipeline = PredictPipeline()

        result = predict_pipeline.predict(
            pred_df
        )


        st.success(
            f"Predicted Price: ₹ {round(result[0], 2)}"
        )


    except Exception as e:

        st.error(e)
