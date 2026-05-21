import os
import sys
import pickle

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    TargetEncoder
)

from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    
    def __init__(self):
        
        self.preprocessor_obj_file_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    
    def get_data_transformer_object(self):
        
        try:

            logging.info("Data transformation initiated")

            
            more_cat_cols = [
                'make',
                'model',
                'city',
                'registrationcity',
                'registrationstate',
                'storename'
            ]

            
            less_cat_cols = [
                'fueltype',
                'bodytype',
                'transmission'
            ]


            transformer = ColumnTransformer(
                transformers=[

                    (
                        'target',
                        TargetEncoder(),
                        more_cat_cols
                    ),

                    (
                        'onehot',
                        OneHotEncoder(
                            handle_unknown='ignore'
                        ),
                        less_cat_cols
                    )

                ],
                remainder='passthrough'
            )

            logging.info("Column transformer created")

            return transformer

        
        except Exception as e:
            
            raise CustomException(e, sys)

    
    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")


            
            train_df['listing_year'] = pd.to_datetime(
                train_df['createdDate']
            ).dt.year

            train_df['car_age'] = (
                train_df['listing_year']
                - train_df['year']
            )


            test_df['listing_year'] = pd.to_datetime(
                test_df['createdDate']
            ).dt.year

            test_df['car_age'] = (
                test_df['listing_year']
                - test_df['year']
            )


            
            columns_to_drop = [
                'name',
                'url',
                'createdDate',
                'listing_year'
            ]

            train_df.drop(
                columns=columns_to_drop,
                inplace=True
            )

            test_df.drop(
                columns=columns_to_drop,
                inplace=True
            )


            
            imputer = SimpleImputer(
                strategy='constant',
                fill_value='Unknown'
            )

            train_df[['bodytype', 'transmission']] = imputer.fit_transform(
                train_df[['bodytype', 'transmission']]
            )

            test_df[['bodytype', 'transmission']] = imputer.transform(
                test_df[['bodytype', 'transmission']]
            )

            logging.info("Missing values handled")


            
            train_df.drop_duplicates(inplace=True)

            logging.info("Duplicates removed")


            
            target_column_name = "price"

            X_train = train_df.drop(
                columns=[target_column_name],
                
            )

            y_train = train_df[target_column_name]

            X_test = test_df.drop(
                columns=[target_column_name],
                
            )

            y_test = test_df[target_column_name]


            
            preprocessing_obj = self.get_data_transformer_object()


            
            X_train_transformed = preprocessing_obj.fit_transform(
                X_train,
                y_train
            )

            X_test_transformed = preprocessing_obj.transform(
                X_test
            )


            
            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train_transformed
            )

            X_test_scaled = scaler.transform(
                X_test_transformed
            )


            
            with open(
                self.preprocessor_obj_file_path,
                "wb"
            ) as file_obj:

                pickle.dump(
                    preprocessing_obj,
                    file_obj
                )

            logging.info("Preprocessor pickle saved")


            
            return (

                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test

            )


        
        except Exception as e:
            
            raise CustomException(e, sys)