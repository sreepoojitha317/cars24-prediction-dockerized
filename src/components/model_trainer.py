import os
import sys
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:

    def __init__(self):

        self.trained_model_file_path = os.path.join(
            "artifacts",
            "xgb_model.pkl"
        )

    
    def evaluate_model(self, y_true, y_pred):

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        return rmse, r2

    
    def initiate_model_trainer(

        self,
        X_train,
        X_test,
        y_train,
        y_test

    ):

        try:

            logging.info("Model training started")


            
            models = {

                "Linear Regression":
                LinearRegression(),

                "Decision Tree":
                DecisionTreeRegressor(
                    max_depth=8
                ),

                "Random Forest":
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                ),

                "XGBoost":
                XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )

            }


            model_report = {}


            
            for model_name, model in models.items():

                logging.info(
                    f"Training started for {model_name}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

                rmse, r2 = self.evaluate_model(
                    y_test,
                    y_pred
                )

                model_report[model_name] = r2

                logging.info(
                    f"{model_name} completed with R2 Score: {r2}"
                )


            
            xgb = XGBRegressor(
                random_state=42
            )


            param_grid = {

                'n_estimators': [100, 200, 300],

                'max_depth': [4, 6, 8],

                'learning_rate': [0.05, 0.1, 0.2],

                'subsample': [0.8, 1],

                'colsample_bytree': [0.8, 1]

            }


            random_search = RandomizedSearchCV(

                estimator=xgb,

                param_distributions=param_grid,

                n_iter=10,

                cv=2,

                verbose=2,

                random_state=42,

                n_jobs=1

            )


            random_search.fit(
                X_train,
                y_train
            )


            best_model = random_search.best_estimator_


            y_pred_best = best_model.predict(
                X_test
            )


            best_rmse, best_r2 = self.evaluate_model(
                y_test,
                y_pred_best
            )


            logging.info(
                f"Best Tuned XGBoost R2 Score: {best_r2}"
            )


            
            joblib.dump(
                best_model,
                self.trained_model_file_path
            )

            logging.info(
                "Best model saved successfully"
            )


            
            return best_r2


        
        except Exception as e:

            raise CustomException(e, sys)