import sys
from src.Credit_Risk_Model.logger import logger

def error_message_detail(error, error_detail:sys):
    """
    Function to handle and log error messages.
    Args:
    error (str): Error message.
    error_detail (sys): Error details.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message[{error}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class for handling exceptions.
    """
    def __init__(self, error_message, error_detail:sys):
        # Use the error_message_detail function to get the full error details
        self.error_message = error_message_detail(error_message, error_detail)
        super().__init__(self.error_message)
        
    def __str__(self):
        return self.error_message
