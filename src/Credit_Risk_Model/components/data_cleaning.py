import os
import sys
import pandas as pd
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import MinMaxScaler
import numpy as np
warnings.filterwarnings('ignore')
from src.Credit_Risk_Model.entity.config_entity import DataCleaningConfig
from src.Credit_Risk_Model.utils.common import read_yaml,create_directories,load_df,save_df
from src.Credit_Risk_Model.constants import *
from src.Credit_Risk_Model.logger import logger

class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.input_filepath = config.input_filepath
        self.test_path = config.test_path
        self.train_path = config.train_path
        self.params = config.params
        
        
    def initiate_train_test_split(self, df):
        try:
            logger.info("Initiating train test split...")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Train test split completed successfully.")
            return train_df, test_df
        except CustomException as e:
            logger.error(f"Error occurred while initiating train test split: {str(e)}")
            raise CustomException(e,sys)
        
    def columns_removal(self, df):
        try:
            logger.info("Removing columns...")
            columns_to_have = self.params.columns_to_have
            df = df.drop(columns_to_have, axis=1)
            logger.info("Columns removed successfully.")
            return df
        except CustomException as e:
            logger.error(f"Error occurred while removing columns: {str(e)}")
            raise CustomException(e,sys)
        
    
    def data_cleaning(self):
        try:
            df = load_df(self.input_filepath)
            train_df, test_df = self.initiate_train_test_split(df)
            logger.info(f"Train_df :{test_df.columns}")
            # Outlier Removal: Processing Fee
            train_df[(train_df.processing_fee / train_df.loan_amount) <= 0.03][["loan_amount","processing_fee"]]
            test_df[(test_df.processing_fee / test_df.loan_amount) <= 0.03][["loan_amount", "processing_fee"]]

            logger.info(f" After outlier removal train_df :{train_df.shape}")
            logger.info(f" After outlier removal test_df :{test_df.shape}")
            
            # Use other business rules for data validation
            # Rule 1: GST should not be more than 20%
            GST = train_df[(train_df.gst/train_df.loan_amount)>0.2].shape
            logger.info(f"Rule 1: GST should not be more than 20% :{GST}")
            
            # Rule 2: Net disbursement should not be higher than loan_amount
            Net = train_df[train_df.net_disbursement>train_df.loan_amount].shape
            logger.info(f"Rule 2: Net disbursement should not be higher than loan_amount :{Net}")
            
            # Fixing Loan Purpose column
            train_df['loan_purpose'] = train_df['loan_purpose'].replace('Personaal', 'Personal')
            test_df['loan_purpose'] = test_df['loan_purpose'].replace('Personaal', 'Personal')
            logger.info(f"Rule 3 : Fixed loan_purpose column :{train_df['loan_purpose'].value_counts()}")
            
            # Generate loan to income (LTI) Ratio
            train_df['loan_to_income'] = round(train_df['loan_amount'] / train_df['income'],2)
            test_df['loan_to_income'] = round(test_df['loan_amount'] / test_df['income'],2)
            logger.info(f"Rule 4 : Generated loan_to_income column(LTI)")

            # Generate Delinquency Ratio
            train_df['delinquency_ratio'] = (train_df['delinquent_months']*100 / train_df['total_loan_months']).round(1)
            test_df['delinquency_ratio'] = (test_df['delinquent_months']*100 / test_df['total_loan_months']).round(1)
            logger.info(f"Generated delinquency_ratio column :{train_df['delinquency_ratio'].describe()}")
            
            # Generate Avg DPD Per Delinquency
            train_df['avg_dpd_per_delinquency'] = np.where(
                    train_df['delinquent_months'] != 0,
                   (train_df['total_dpd'] / train_df['delinquent_months']).round(1),
                    0
                )

            test_df['avg_dpd_per_delinquency'] = np.where(
                    test_df['delinquent_months'] != 0,
                    (test_df['total_dpd'] / test_df['delinquent_months']).round(1),
                        0
                )
            
            train_df = train_df.drop(['cust_id', 'loan_id'],axis="columns")
            test_df = test_df.drop(['cust_id', 'loan_id'],axis="columns")
            
            # Remove columns that business contact person asked us to remove
            train_df = self.columns_removal(train_df)
            test_df = self.columns_removal(test_df)
            
            save_df(file_path=self.train_path,df=train_df)
            save_df(file_path=self.test_path,df=test_df)
            
           
            
        except CustomException as e:
            logger.error(f"Error occurred while data cleaning: {str(e)}")
            raise CustomException(e,sys)
