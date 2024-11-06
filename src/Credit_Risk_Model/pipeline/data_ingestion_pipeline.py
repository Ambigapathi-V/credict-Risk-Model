from src.Credit_Risk_Model.config.configuration import ConfigurationManager
from src.Credit_Risk_Model.components.data_ingestion import DataIngestion
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import os
import sys
STAGE_NAME = 'Data Ingestion Stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self):
        
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.load_data()
            
        except CustomException as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            return CustomException(e,sys)

    
if __name__ == "__main__":
    try: 
        pipeline = DataIngestionTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_ingestion()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)