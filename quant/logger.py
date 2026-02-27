import logging
import os
from quant.config import CONF

def setup_logger(name: str = "quant") -> logging.Logger:
    """Setup and configure a production-grade logger."""
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times in interactive sessions
    if logger.hasHandlers():
        return logger

    level = getattr(logging, CONF.log.level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_file = CONF.log.file
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# Global configured logger
logger = setup_logger()
