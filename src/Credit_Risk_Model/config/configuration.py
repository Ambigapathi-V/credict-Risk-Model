from src.Credit_Risk_Model.constants import *
from src.Credit_Risk_Model.utils.common import read_yaml,create_directories,load_df,save_df
from  src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.entity.config_entity import (DataIngestionConfig)

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            customers_data=Path(config.customers_data),
            loans_data=Path(config.loans_data),
            bureau_data=Path(config.bureau_data),
            output_path=Path(config.output_path)
        )
        return data_ingestion_config
                           