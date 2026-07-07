# utils/logger_setup.py
import logging
import os
from datetime import datetime


def setup_logging(log_dir, log_filename_prefix, level=logging.INFO):
    """
    Sets up a logger with a file handler and a console handler.
    Prevents adding duplicate handlers if called multiple times for the same logger.

    Args:
        log_dir (str): The directory where log files will be saved.
        log_filename_prefix (str): A prefix for the log filename.
                                   The final filename will be e.g., 'prefix_YYYYMMDD_HHMMSS.log'.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG, logging.WARNING).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate unique log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = os.path.join(log_dir, f"{log_filename_prefix}_{timestamp}.log")

    # Get a logger instance. Using the prefix as the logger name is good practice.
    # This ensures that if setup_logging is called multiple times with the same prefix,
    # it configures the same logger instance.
    logger = logging.getLogger(log_filename_prefix)

    # Prevent adding handlers multiple times if the logger has already been configured
    if logger.handlers:
        for handler in logger.handlers[:]:  # Iterate on a copy of the list
            logger.removeHandler(handler)

    logger.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger