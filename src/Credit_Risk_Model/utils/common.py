import os
import sys
import yaml
from src.Credit_Risk_Model.logger import logger
from src.Credit_Risk_Model.exception import CustomException
import joblib
import pandas as pd
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import List, Dict
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    """
    Reads a yaml file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the yaml file.
    
    Raises:
        ValueError: if yaml file is empty or malformed
    
    Returns:
        ConfigBox: A ConfigBox object containing the parsed data.
    """
    try:
        with open(path_to_yaml) as yamlfile:
            content = yaml.safe_load(yamlfile)
            logger.info(f"Successfully loaded yaml file: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error reading yaml file: {path_to_yaml} - {str(e)}")
        raise ValueError(f"Error reading yaml file: {path_to_yaml}")
    except Exception as e:
        logger.error(f"Error reading yaml file: {path_to_yaml} - {str(e)}")
        raise CustomException(e,sys)
        
        

@ensure_annotations
def create_directories (path_to_directories: list , verbose = True):
    """
    Creates directories in the given path if they don't exist.
    
    Args:
        path_to_directories (list): A list of directories to be created.
        verbose (bool): A flag to indicate whether to print info messages.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")

@ensure_annotations
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)
            
    except Exception as e:
        logger.error(f"Error saving object: {file_path} - {str(e)}")
        raise CustomException(e, sys)
    
    
@ensure_annotations
def load_df(file_path : Path) -> pd.DataFrame:
    """
    Loads a DataFrame from a csv file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded DataFrame from: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from: {file_path} - {str(e)}")
        raise CustomException(e, sys)
    
@ensure_annotations
def save_df (file_path : Path, df : pd.DataFrame):
    """
    Saves a DataFrame to a csv file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        df.to_csv(file_path, index=False)
        
        logger.info(f"Successfully saved DataFrame to: {file_path}")
            
    except Exception as e:
        logger.error(f"Error saving DataFrame to: {file_path} - {str(e)}")
        raise CustomException(e, sys)