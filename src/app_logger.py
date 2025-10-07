import os
from datetime import datetime

from .app_config import AppConfig


class CSVLogger:
    """ 
    CSV Logger class for logging data to a CSV file.
    """
    def __init__(self, log_name: str, run_id: (str | None) = None):
        """
        Initializes the CSVLogger.
        params:
            log_name (str): The name of the log file. Must be a valid file name.
            run_id (str, optional): An identifier for the run. If not provided, defaults to None.
        raises:
            ValueError: If log_name is not a string, is empty, or is not a valid file name.
        """
        if not isinstance(log_name, str):
            raise ValueError("log_name must be a string")
        if not log_name:
            raise ValueError("log_name cannot be an empty string")
        if not log_name.isidentifier():
            raise ValueError("log_name must be a valid identifier (alphanumeric and underscores only)")
        self.file = None
        self.log_name = log_name
        self.run_id = run_id
        
    def __enter__(self):
        """ 
        Opens the log file for writing. If the directory does not exist, it creates it.
        params:
            None
        returns:
            self: Returns the instance of the CSVLogger.
        raises:
            RuntimeError: If the logger is already open.
            ValueError: If log_path is not set in AppConfig or is not a string.
            Exception: If there is an error opening the log file.
        """
        if self.file is not None:
            raise RuntimeError("Logger is already open.")
        if not AppConfig.log_path:
            raise ValueError("Log path is not set in AppConfig.")
        if not isinstance(AppConfig.log_path, str):
            raise ValueError("log_path in AppConfig must be a string")
        try:
            dir = os.path.dirname(__file__)
            if not os.path.exists(AppConfig.log_path):
                os.makedirs(AppConfig.log_path, exist_ok=True)
            self.file = open(f"{AppConfig.log_path}/{self.log_name}_{self.run_id}.csv", "a")
        except Exception as e:
            print(f"Error opening log file: {e}")
            raise e
        return self

    def log(self, data):
        """ 
        Logs data to the CSV file.
        params:
            data (str, int, float): The data to log. Must be a string, integer, or float.
        raises:
            RuntimeError: If the logger is not open.
            ValueError: If data is not a string, integer, or float, or if it is empty.
            Exception: If there is an error converting data to a string.
        """
        if self.file is None:
            raise RuntimeError("Logger is not open.")
        if not isinstance(data, (str, int, float)) or (isinstance(data, str) and not data.strip()) or (not data):
            raise ValueError("Data must be a string, integer, or float and cannot be empty. Unsupported type: {}".format(type(data)))
        if isinstance(data, (int, float)):
            try:
                data = str(data)
            except Exception as e:
                raise ValueError("Data must be a string, integer, or float. Unsupported type: {}".format(type(data)))
      
        self.file.write(data + '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ 
        Closes the log file.
        
        """
        if self.file is not None:
            self.file.close()