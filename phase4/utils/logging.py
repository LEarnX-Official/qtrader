import logging, sys
from phase4.utils.config import LOGS_P4

def get_logger(name): return logging.getLogger(name)

def setup_logging(level=logging.INFO):
    LOGS_P4.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(LOGS_P4/"phase4.log", mode="a")])
