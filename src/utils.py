import os
import logging

def setup_logging(log_path="outputs/logs/extraction.log"):
    """
    Sets up basic logging to file and console.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_dir(path):
    """
    Ensures a directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
