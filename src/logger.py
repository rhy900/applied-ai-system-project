import logging
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def setup_logger(name: str = "music_agent") -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "agent.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_guardrail(logger: logging.Logger, reason: str) -> None:
    logger.warning(f"GUARDRAIL: {reason}")
