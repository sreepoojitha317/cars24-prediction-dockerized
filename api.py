from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)


app = FastAPI()


class CarData(BaseModel):

    make: str
    model: str
    year: int
    fueltype: str
    kilometerdriven: int
    ownernumber: int
    transmission: str
    bodytype: str
    registrationcity: str
    registrationstate: str
    city: str
    storename: str
    isc24assured: bool
    benefits: float
    discountprice: float
    car_age: int


@app.get("/")

def home():

    return {
        "message": "Cars24 Price Prediction API Running"
    }


@app.post("/predict")

def predict(data: CarData):

    custom_data = CustomData(

        make=data.make,

        model=data.model,

        year=data.year,

        fueltype=data.fueltype,

        kilometerdriven=data.kilometerdriven,

        ownernumber=data.ownernumber,

        transmission=data.transmission,

        bodytype=data.bodytype,

        registrationcity=data.registrationcity,

        registrationstate=data.registrationstate,

        city=data.city,

        storename=data.storename,

        isc24assured=data.isc24assured,

        benefits=data.benefits,

        discountprice=data.discountprice,

        car_age=data.car_age

    )


    pred_df = custom_data.get_data_as_dataframe()


    predict_pipeline = PredictPipeline()

    result = predict_pipeline.predict(
        pred_df
    )


    return {

        "predicted_price": float(result[0])

    }