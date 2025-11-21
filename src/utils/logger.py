import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "output/logs", level=logging.INFO):
    """
    Sets up a logger that writes both to console and a rotating file.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to save logs
        level (int): Logging level (default INFO)
        
    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File Handler
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger
