from dotenv import load_dotenv
import logging
from logging.handlers import TimedRotatingFileHandler
import os

load_dotenv()

logging.getLogger().setLevel(os.getenv('LOG_LEVEL'))

def get_logger(name):
    log_level = os.getenv('LOG_LEVEL')
    log_dir = os.getenv('LOG_DIR')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # create and configure main logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create console handler
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)
    # create formatter and add it to the handler
    formatter_stream = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stream.setFormatter(formatter_stream)
    # add the handler to the logger
    logger.addHandler(handler_stream)
    handler_file = TimedRotatingFileHandler(f'{log_dir}/{name}.log', 'midnight')
    handler_file.setLevel(logging.INFO)
    # create formatter and add it to the handler
    formatter_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_file.setFormatter(formatter_file)
    # add the handler to the logger
    logger.addHandler(handler_file)
    return logger

def get_logger_child(name):
    return logging.getLogger(name)


