import os
import sys
import logging

logging_str = "[%(asctime)s] %(levelname)s: %(module)s : %(lineno)d] %(message)s"

# Set up logging
logg_dir = "logs"
log_filepath = os.path.join(logg_dir, "logging.log")
os.makedirs(logg_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers =[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)  # Output to console as well
    ]
)


# Initialize the logger
logger = logging.getLogger("Credit_Risk_Model")

# Log a startup message
logger.info("Credit Risk Model Building Started")