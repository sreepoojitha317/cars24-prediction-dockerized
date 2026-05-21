import os
import sys
import pandas as pd
import joblib

from src.exception import CustomException


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "xgb_model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )


    def predict(self, features):

        try:

            model = joblib.load(
                self.model_path
            )

            preprocessor = joblib.load(
                self.preprocessor_path
            )

            data_scaled = preprocessor.transform(
                features
            )

            preds = model.predict(
                data_scaled
            )

            return preds


        except Exception as e:

            raise CustomException(e, sys)



class CustomData:

    def __init__(

        self,

        make,
        model,
        year,
        fueltype,
        kilometerdriven,
        ownernumber,
        transmission,
        bodytype,
        registrationcity,
        registrationstate,
        city,
        storename,
        isc24assured,
        benefits,
        discountprice,
        car_age

    ):

        self.make = make
        self.model = model
        self.year = year
        self.fueltype = fueltype
        self.kilometerdriven = kilometerdriven
        self.ownernumber = ownernumber
        self.transmission = transmission
        self.bodytype = bodytype
        self.registrationcity = registrationcity
        self.registrationstate = registrationstate
        self.city = city
        self.storename = storename
        self.isc24assured = isc24assured
        self.benefits = benefits
        self.discountprice = discountprice
        self.car_age = car_age


    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "make": [self.make],

                "model": [self.model],

                "year": [self.year],

                "fueltype": [self.fueltype],

                "kilometerdriven": [self.kilometerdriven],

                "ownernumber": [self.ownernumber],

                "transmission": [self.transmission],

                "bodytype": [self.bodytype],

                "registrationcity": [
                    self.registrationcity
                ],

                "registrationstate": [
                    self.registrationstate
                ],

                "city": [self.city],

                "storename": [self.storename],

                "isc24assured": [
                    self.isc24assured
                ],

                "benefits": [self.benefits],

                "discountprice": [
                    self.discountprice
                ],

                "car_age": [self.car_age]

            }

            return pd.DataFrame(
                custom_data_input_dict
            )


        except Exception as e:

            raise CustomException(e, sys)