import logging
import sys


def make_stderr_handler(level):
    date_fmt = '%Y-%m-%d %H:%M:%S %z'
    fmt = '%(levelname)s %(asctime)s %(name)s %(message)s'
    formatter = logging.Formatter(fmt, date_fmt)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def make_logger(name: str, level: str = 'info'):
    logger = logging.getLogger(name)
    logger.propagate = False

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    logger.addHandler(make_stderr_handler(log_level))

    return logger
