import os
import sys
import pandas as pd
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from src.Credit_Risk_Model.entity.config_entity import DataCleaningConfig
from src.Credit_Risk_Model.utils.common import read_yaml, create_directories, load_df, save_df, save_object,save_numpy_array_data
from src.Credit_Risk_Model.constants import *
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.constants import PREPROCESSOR

class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.input_filepath = config.input_filepath
        self.test_path = config.test_path
        self.train_path = config.train_path
        self.target_column = config.target_column
        self.columns_to_have = config.columns_to_have
        self.model_path = config.model_path
        self.columns = config.columns
        self.columns_dtypes = config.columns_dtypes

    def initiate_train_test_split(self, df):
        try:
            logger.info("Initiating train test split...")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Train test split completed successfully.")
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error occurred while initiating train test split: {str(e)}")
            raise CustomException(str(e), sys)

    def columns_removal(self, df):
        try:
            logger.info("Removing unnecessary columns...")
            df = df[self.columns_to_have]
            logger.info("Columns removed successfully.")
            return df
        except Exception as e:
            logger.error(f"Error occurred while removing columns: {str(e)}")
            raise CustomException(str(e), sys)

    def checking_schema(self, df):
        try:
            # Validate columns
            missing_columns = set(self.columns) - set(df.columns)
            extra_columns = set(df.columns) - set(self.columns)

            if missing_columns:
                logger.error(f"Missing columns: {', '.join(missing_columns)}")
            if extra_columns:
                logger.error(f"Extra columns: {', '.join(extra_columns)}")

            if missing_columns or extra_columns:
                return False

            logger.info("Schema validation successful")
            return True
        except Exception as e:
            logger.error(f"Error occurred while checking schema: {str(e)}")
            raise CustomException(str(e), sys)

    def apply_business_rules(self, train_df, test_df):
        try:
            logger.info("Applying business rules...")
            # Rule 1: Processing Fee should not be more than 3% of loan amount
            train_df = train_df[(train_df.processing_fee / train_df.loan_amount) <= 0.03]
            test_df = test_df[(test_df.processing_fee / test_df.loan_amount) <= 0.03]
            logger.info(f"After applying Processing Fee rule, train_df: {train_df.shape}, test_df: {test_df.shape}")

            # Rule 2: GST should not be more than 20%
            gst_violation_train = train_df[(train_df.gst / train_df.loan_amount) > 0.2]
            logger.info(f"GST Violations in train data: {gst_violation_train.shape[0]} records")

            # Rule 3: Net disbursement should not be higher than loan amount
            net_disbursement_violation_train = train_df[train_df.net_disbursement > train_df.loan_amount]
            logger.info(f"Net Disbursement Violations in train data: {net_disbursement_violation_train.shape[0]} records")

            # Rule 4: Fixing loan_purpose column spelling errors
            train_df['loan_purpose'] = train_df['loan_purpose'].replace('Personaal', 'Personal')
            test_df['loan_purpose'] = test_df['loan_purpose'].replace('Personaal', 'Personal')
            logger.info(f"Fixed loan_purpose column: {train_df['loan_purpose'].value_counts()}")

            # Additional rules for generating ratios
            train_df['loan_to_income'] = round(train_df['loan_amount'] / train_df['income'], 2)
            test_df['loan_to_income'] = round(test_df['loan_amount'] / test_df['income'], 2)
            logger.info(f"Generated loan_to_income column")

            # Generate delinquency ratio
            train_df['delinquency_ratio'] = (train_df['delinquent_months'] * 100 / train_df['total_loan_months']).round(1)
            test_df['delinquency_ratio'] = (test_df['delinquent_months'] * 100 / test_df['total_loan_months']).round(1)
            logger.info(f"Generated delinquency_ratio column: {train_df['delinquency_ratio'].describe()}")

            # Generate avg_dpd_per_delinquency
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
            logger.info(f"Generated avg_dpd_per_delinquency column: {train_df['avg_dpd_per_delinquency'].describe()}")

            return train_df, test_df
        except Exception as e:
            logger.error(f"Error occurred while applying business rules: {str(e)}")
            raise CustomException(str(e), sys)

    def data_cleaning(self):
        try:
            df = load_df(self.input_filepath)
            train_df, test_df = self.initiate_train_test_split(df)

            # Apply business rules to data
            train_df, test_df = self.apply_business_rules(train_df, test_df)
            train_target = train_df[self.target_column]
            test_target = test_df[self.target_column]
            
             # Removing columns that aren't required
            train_df = self.columns_removal(train_df)
            test_df = self.columns_removal(test_df)
            

            # Validate Schema
            if not self.checking_schema(train_df):
                raise CustomException("Schema validation failed for train data.")
            if not self.checking_schema(test_df):
                raise CustomException("Schema validation failed for test data.")



            # Prepare preprocessor pipeline for numerical and categorical features
            logger.info(f"{train_df.columns}")
            categorical_columns = train_df.select_dtypes(include=['object']).columns
            logger.info(f"Categorical columns: {categorical_columns}")
            numerical_columns = train_df.select_dtypes(exclude=['object']).columns
            logger.info(f"Numerical columns: {numerical_columns}")
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=False))])
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder()),
            ])

            preprocessor = ColumnTransformer(
                transformers=[('num', numerical_transformer, numerical_columns),
                              ('cat', categorical_transformer, categorical_columns)]
            )

            # Transform train and test data
            train_df = preprocessor.fit_transform(train_df)
            test_df = preprocessor.transform(test_df)

            # Save the preprocessor object for later use
            save_object(file_path=self.model_path, obj=preprocessor)
            save_object(file_path='model/preprocessor.joblib', obj=preprocessor)

            train_arr = np.c_[train_df, np.array(train_target)]
            test_arr = np.c_[test_df, np.array(test_target)]
            
            # Save the numpy arrays
            save_numpy_array_data(file_path=self.train_path, array=train_arr)
            save_numpy_array_data(file_path=self.test_path, array=test_arr)

            logger.info("Data cleaning process completed successfully.")

        except Exception as e:
            logger.error(f"Error occurred during data cleaning: {str(e)}")
            raise CustomException(str(e), sys)
