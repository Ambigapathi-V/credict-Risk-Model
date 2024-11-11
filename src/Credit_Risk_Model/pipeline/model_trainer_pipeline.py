from src.Credit_Risk_Model.config.configuration import ConfigurationManager
from src.Credit_Risk_Model.components.model_trainer import ModelTrainer
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import os
import sys
STAGE_NAME = 'Model Trainer Stage'

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_training(self):
        try:
            
            config = ConfigurationManager()
            model_training_config = config.get_model_training_config()
            preprocessing = ModelTrainer(model_training_config)
            preprocessing.initiate_model_trainer()
            
        except CustomException as e:
            logger.error(f"Error Occurred: {e}")
            raise CustomException(e, sys)
        
        
# Running the Data Pre-Processing Training Pipeline
if __name__ == "__main__":
    try: 
        pipeline = ModelTrainerTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_model_training()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)