from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DataIngestionConfig:
    root_dir: Path
    customers_data: Path
    loans_data: Path
    bureau_data: Path
    output_path: Path    

@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    input_data: Path
    output_data: Path

@dataclass
class DataCleaningConfig:
    root_dir: Path
    input_filepath: Path
    test_path: Path
    train_path: Path
    columns_to_have: List[str]
    model_path: Path  
    target_columns : str


@dataclass
class DataIngestionConfig:
    root_dir: Path
    customers_data: Path
    loans_data: Path
    bureau_data: Path
    output_path: Path    

@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    input_data: Path
    output_data: Path

@dataclass
class DataCleaningConfig:
    root_dir: Path
    input_filepath: Path
    test_path: Path
    train_path: Path
    columns_to_have: List[str]
    model_path: Path  
    target_column : str
    columns : List[str]
    columns_dtypes : dict[str:str]

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    test_path: Path
    train_path: Path
    model_path: Path
    preprocessor : Path
    
    
    
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score:float
    recall_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path : str
    train_metric_artifact : ClassificationMetricArtifact
    test_metric_artifact : ClassificationMetricArtifact
