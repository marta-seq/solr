# src/utils/file_helpers.py
import os
import logging
from datetime import datetime # Import datetime for get_log_file_path

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """
    Ensures that a directory exists. If it does not, it creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_log_file_path(log_dir, prefix):
    """
    Generates a unique log file path with a timestamp.
    """
    ensure_dir(log_dir) # Ensure log directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{prefix}_{timestamp}.log")