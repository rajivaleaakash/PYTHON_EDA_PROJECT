import logging
from pathlib import Path
import os

def setup_logger(script_name):
    try:
        # create 'logs' directory if it doesn't exist
        log_dir = Path(os.getcwd())/"logs"
        log_dir.mkdir(exist_ok=True)

        # Define log file path
        log_file = log_dir/f"{script_name}.log"

        # create logger with the given name
        logger = logging.getLogger(script_name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # file handler for saving logs
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            # console handler for real-time output
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Define a consistent log format
            formatter = logging.Formatter(
                fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
                datefmt="%y-%m-%d %H:%M:%S"
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Attach handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    except Exception as e:
        print(f"[Error] Failed to set up logger: {e}")