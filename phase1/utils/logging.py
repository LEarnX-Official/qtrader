import logging
import sys
from pathlib import Path

from phase1.utils.config import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "phase1.log", mode="a"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
