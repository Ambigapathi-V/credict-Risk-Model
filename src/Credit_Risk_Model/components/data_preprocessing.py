import os
import sys
import pandas as pd
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import warnings
warnings.filterwarnings('ignore')
from src.Credit_Risk_Model.entity.config_entity import DataPreprocessingConfig
from src.Credit_Risk_Model.utils.common import read_yaml,create_directories,load_df,save_df





class DataPreprocessing:
    def __init__(self, config :DataPreprocessingConfig):
        self.input_data = config.input_data
        self.output_data = config.output_data
        
    def data_preprocessing(self):
        try:
            df = load_df(self.input_data)
            
            # Makethe 'Default' column  into "int"
            df['default'] = df['default'].astype(int)
            logger.info(f"The Valure counts for 'default' column: {df['default'].value_counts()}")
            
            ## Check for missing values
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype == 'object':
                        # Fill missing values with 'Unknown'for categorical columns
                        logger.warning(f"Missing values found in column '{col}', replacing with 'Unknown'")
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        logger.info(f"Missing values replaced in column '{col}' with mode '{df[col].mode()[0]}'")
                    else:
                        # Fill missing values with mean for numerical columns
                        logger.warning(f"Missing values found in column '{col}', replacing with mean")
                        df[col].fillna(df[col].mean(), inplace=True)
                        logger.info(f"Missing values replaced in column '{col}' with mean '{df[col].mean()}'")
                else:
                    logger.info(f"No missing values found in column '{col}'")

            ## Check the Duplicate rows
            if df.duplicated().sum() > 0:
                logger.warning(f"Duplicate rows found, dropping them")
                df.drop_duplicates(inplace=True)
                logger.info(f"Duplicate rows dropped")
                
            else:
                logger.info(f"No duplicate rows found")
                
            logger.info(df.columns)
            
            # Save the processed data
            save_df(df=df, file_path=self.output_data)

        
        except CustomException as e:
            logger.error(f"An error occurred during data preprocessing: {str(e)}")
            raise CustomException(e,sys)
            
    
        
        