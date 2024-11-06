import os
import pandas as pd
from  src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.constants import *
from src.Credit_Risk_Model.utils.common import read_yaml,create_directories,load_df,save_df
from  src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.customers_data = config.customers_data
        self.loans_data = config.loans_data
        self.bureau_data = config.bureau_data
        self.output_path = config.output_path
    
    def load_data(self):
        logger.info("Loading customers data...")
        customers_df = load_df(self.customers_data)
        
        logger.info("Loading loans data...")
        loans_df = load_df(self.loans_data)
        
        logger.info("Loading bureau data...")
        bureau_df = load_df(self.bureau_data)
        
        logger.info(f"The Shape of customers data: {customers_df.shape}")
        logger.info(f"The Shape of loans data: {loans_df.shape}")
        logger.info(f"The Shape of bureau data: {bureau_df.shape}")
        
        df = pd.merge(customers_df, loans_df, on='cust_id')
        df = pd.merge(df, bureau_df, on='cust_id')
        
        save_df(df=df, file_path=self.output_path)        
    