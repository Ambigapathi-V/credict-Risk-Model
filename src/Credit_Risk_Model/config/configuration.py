from src.Credit_Risk_Model.constants import *
from src.Credit_Risk_Model.utils.common import read_yaml, create_directories, load_df, save_df
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig,DataCleaningConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        # Read YAML files for configuration, params, and schema
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        # Create the root directories specified in the config
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        This method loads the data ingestion configuration from the YAML file
        and returns a DataIngestionConfig object.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        # Create and return a DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            customers_data=Path(config.customers_data),
            loans_data=Path(config.loans_data),
            bureau_data=Path(config.bureau_data),
            output_path=Path(config.output_path)
        )
        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        This method loads the data preprocessing configuration from the YAML file
        and returns a DataPreprocessingConfig object.
        """
        config = self.config.data_preprocessing
        create_directories([config.root_dir])
        
        # Create and return a DataPreprocessingConfig object
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            input_data=Path(config.input_data),
            output_data=Path(config.output_data)
        )
        
        return data_preprocessing_config
    
    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config['data_cleaning']
        params = self.params['data_cleaning']
        create_directories([config.root_dir])
        
        data_cleaning_config = DataCleaningConfig(
            root_dir = Path(config.root_dir),
            input_filepath = Path(config.input_filepath),
            test_path = Path(config.test_path),
            train_path = Path(config.train_path),
            columns_to_have = params.columns_to_have,
            params = params,
        )
        return data_cleaning_config
