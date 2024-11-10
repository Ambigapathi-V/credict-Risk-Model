from dataclasses import dataclass
from pathlib import Path
from typing import List
@dataclass

class DataIngestionConfig:
    root_dir : Path
    customers_data : Path
    loans_data : Path
    bureau_data : Path
    output_path : Path    
    
@dataclass
class DataPreprocessingConfig:
    root_dir : Path
    input_data : Path
    output_data : Path
    
@dataclass
class DataCleaningConfig:
    root_dir: Path
    input_filepath : Path
    test_path : Path
    train_path : Path
    columns_to_have : List[str]
    params : dict