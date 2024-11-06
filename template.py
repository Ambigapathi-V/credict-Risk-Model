import os
from pathlib import Path
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Project-specific variables
project_name = "Credit_Risk_Model"

# List of files and directories to create
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/exception/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "Dockerfile",
    "setup.py",
    "research/research.ipynb",
    "template/index.html",
    "requirements.txt",
    "app.py",
    "api.py",
    "README.md",
    ".gitignore",
]

# Create files and directories
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    # Create directory if it doesn't exist
    if filedir != "":
        try:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory {filedir} for the file {filename}")
        except Exception as e:
            logging.error(f"Failed to create directory {filedir}: {e}")
    
    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        try:
            with open(filepath, "w") as f:
                pass
            logging.info(f"Created empty file {filepath}")
        except Exception as e:
            logging.error(f"Failed to create file {filepath}: {e}")
    else:
        logging.info(f"File {filepath} already exists")
