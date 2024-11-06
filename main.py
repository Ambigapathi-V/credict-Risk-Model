from src.Credit_Risk_Model.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import sys

STAGE_NAME = "Data Ingestion Stage"

try: 
        pipeline = DataIngestionTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_ingestion()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)