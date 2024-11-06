from src.Credit_Risk_Model.config.configuration import ConfigurationManager
from src.Credit_Risk_Model.components.data_preprocessing import DataPreprocessing
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import os
import sys
STAGE_NAME = 'Data Pre-Processing Stage'

class DataPreProcessingTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_preprocessing(self):
        try:
            
            config = ConfigurationManager()
            data_preprocessing_config = config.get_data_preprocessing_config()
            preprocessing = DataPreprocessing(data_preprocessing_config)
            preprocessing.data_preprocessing()
            
        except CustomException as e:
            logger.error(f"Error Occurred: {e}")
            raise CustomException(e, sys)
        
        
# Running the Data Pre-Processing Training Pipeline
if __name__ == "__main__":
    try: 
        pipeline = DataPreProcessingTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_preprocessing()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)