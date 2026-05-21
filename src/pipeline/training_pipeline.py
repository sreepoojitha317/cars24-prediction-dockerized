import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer



class TrainingPipeline:

    def start_training_pipeline(self):

        try:

            logging.info(
                "Training pipeline started"
            )


            
            data_ingestion = DataIngestion()

            train_data_path, test_data_path = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                "Data ingestion completed"
            )


            
            data_transformation = DataTransformation()

            (
                X_train,
                X_test,
                y_train,
                y_test

            ) = data_transformation.initiate_data_transformation(

                train_data_path,
                test_data_path

            )

            logging.info(
                "Data transformation completed"
            )


            
            model_trainer = ModelTrainer()

            r2_score = model_trainer.initiate_model_trainer(

                X_train,
                X_test,
                y_train,
                y_test

            )

            logging.info(
                f"Model training completed with R2 Score: {r2_score}"
            )

            print(
                f"Training completed successfully. R2 Score: {r2_score}"
            )


        
        except Exception as e:

            raise CustomException(e, sys)