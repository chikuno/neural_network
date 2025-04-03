# config/logging_config.py

import logging

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name, log file, and logging level.
    Returns the configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

# Example usage:
# training_logger = setup_logger('training', 'logs/training.log')
# data_logger = setup_logger('data_scraping', 'logs/data_scraping.log')
