# src/utils/logging_setup.py
import logging
import os
from datetime import datetime
import sys

# Add project root to path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'review' is the actual project root containing 'src'
project_root = "/home/martinha/PycharmProjects/phd/review"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def setup_logging(log_dir: str = "data/logs/", log_prefix: str = "pipeline_log", level=logging.DEBUG, disable_file_logging=False):
    """
    Sets up logging to console and a dated file.
    Logs will be saved in the specified log_dir.
    Can disable file logging for tests.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    # Clear existing handlers to prevent duplicate logs during multiple test runs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(level) # Set the overall level

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # File handler (optional, for pipeline runs)
    if not disable_file_logging:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging configured. Log file: {log_file}")
    else:
        root_logger.info("File logging disabled for this run (e.g., for unit tests).")


    # Set specific levels for potentially noisy loggers if needed
    logging.getLogger('Bio').setLevel(logging.WARNING) # BioPython can be chatty
    logging.getLogger('requests').setLevel(logging.WARNING) # Requests can be chatty
    logging.getLogger('urllib3').setLevel(logging.WARNING) # urllib3 can be chatty

    # Ensure loggers from other modules also respect the root level
    # This is often managed by setting the root logger level, but explicit checks can help
    # For example, if you want specific modules to be less verbose during DEBUG
    # logging.getLogger('src.utils.data_helpers').setLevel(logging.INFO)
    # logging.getLogger('src.weak_annotation.concept_extractor').setLevel(logging.INFO)

if __name__ == "__main__":
    setup_logging(level=logging.INFO) # Default to INFO for direct execution
    logger = logging.getLogger(__name__)
    logger.info("This is an INFO message from logging_setup.py")
    logger.debug("This is a DEBUG message (will only show if level is DEBUG)")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
