from dataclasses import dataclass
from pathlib import Path
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