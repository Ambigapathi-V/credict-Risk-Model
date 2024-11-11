from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
from src.Credit_Risk_Model.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Credit_Risk_Model.pipeline.data_preprocessing_pipeline import DataPreProcessingTrainingPipeline
from src.Credit_Risk_Model.pipeline.data_cleaning_pipeline import DataCleaningTrainingPipeline
from src.Credit_Risk_Model.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
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

STAGE_NAME = "Data Pre-Processing Stage"

try: 
        pipeline = DataPreProcessingTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_preprocessing()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)

STAGE_NAME = "Data Cleaning Stage"

try: 
        pipeline = DataCleaningTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_cleaning()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)

STAGE_NAME = "Model Trainer Stage"

try: 
        pipeline = ModelTrainerTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_model_training()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)

