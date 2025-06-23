from functools import wraps
import logging
from os import path
import os
from pathlib import Path
import warnings
import errno


def get_logger(file_name: str, folder_path: str = None, logger_system_name: str = 'logger'):
    file_name += '.log'
    if folder_path != None:
        if path.exists(folder_path) == False:
            warnings.warn(
                f"folder {folder_path} don't exist. The creation process has started.")
            try:
                os.makedirs(folder_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif os.path.isdir(folder_path) == False:
            raise ValueError(f"{folder_path} is not a folder")
    else:
        folder_path = '.'
    log_file_path = Path(folder_path) / Path(file_name)
    # config loger
    logger = logging.getLogger('Arcrobov_V_1_logger')
    logger.setLevel(logging.INFO)

    # set file pth to log
    handler = logging.FileHandler(log_file_path)
    logger.addHandler(handler)
    formatter = logging.Formatter(
        '%(message)s - %(filename)s - %(lineno)d - %(pathname)s - %(process)d - %(processName)s - %(thread)d - %(threadName)s')
    handler.setFormatter(formatter)
    logger.info('info about user:')
    # set formet for log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    return logger


def log_func(logger: logging.Logger, result):
    logger.info(result)


def log_to_decorator(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.info(f"Function {func.__name__} returned: {result}")
            return result
        return wrapper
    return decorator
