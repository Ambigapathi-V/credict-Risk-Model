from src.Credit_Risk_Model.config.configuration import ConfigurationManager
from src.Credit_Risk_Model.components.data_cleaning import DataCleaning
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import os
import sys
STAGE_NAME = 'Data cleaning Stage'

class DataCleaningTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_cleaning(self):
        
        try:
            config = ConfigurationManager()
            data_cleaning = config.get_data_cleaning_config()
            data_cleaner = DataCleaning(data_cleaning)
            cleaned_data = data_cleaner.data_cleaning()
            
        except CustomException as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            return CustomException(e,sys)

    
if __name__ == "__main__":
    try: 
        pipeline = DataCleaningTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_cleaning()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)