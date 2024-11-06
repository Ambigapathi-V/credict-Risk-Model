import sys
from src.Credit_Risk_Model.logger import logger

def error_message_detail(error, error_detail:sys):
    """
    Function to handle and log error messages.
    Args:
    error (str): Error message.
    error_detail (sys): Error details.
    """
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message_detail  = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message[{error}]"
    return error_message_detail

class CustomException(Exception):
    """
    Custom exception class for handling exceptions.
    """
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        return self.error_message