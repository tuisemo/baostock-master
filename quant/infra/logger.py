import logging
import os
import sys
from quant.infra.config import CONF

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

    # Windows 默认控制台编码可能无法输出 emoji 等字符，导致 UnicodeEncodeError。
    # 使用 replace 避免日志输出影响主流程（尤其是实盘/自动化任务）。
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

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
