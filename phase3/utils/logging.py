import logging, sys
from phase3.utils.config import LOGS_P3

def get_logger(name): return logging.getLogger(name)

def setup_logging(level=logging.INFO):
    LOGS_P3.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(LOGS_P3/"phase3.log", mode="a")])
