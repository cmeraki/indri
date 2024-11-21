import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "line_number": record.lineno,
            "message": record.getMessage(),
            "thread_id": record.thread,
            "process_id": record.process
        }

        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)

        return json.dumps(log_obj)

def get_logger(name: str, level: str = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove all handlers associated with the logger object
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = JsonFormatter(datefmt='%Y-%m-%dT%H:%M:%S.%fZ')

    log_file = f"inference.jsonl"
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", log_file)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(thread)d | %(process)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.info("This is an info message with extra fields", extra={"user_id": "123", "ip_address": "192.168.1.1", "action": "login"})

    try:
        raise ValueError("Something went wrong")
    except Exception as e:
        logger.exception("An error occurred")
