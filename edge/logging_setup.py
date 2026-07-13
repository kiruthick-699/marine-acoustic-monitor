"""
Central logging configuration for edge/.

One place that wires up the "edge" logger tree (edge.main, edge.calibration,
edge.scheduler, edge.capture_loop, ...) with a rotating file handler --
writing to config.storage.log_path, so an unattended systemd-run process
(docs/pi-implementation.md) has a bounded on-disk log -- and a console
handler, so interactive runs (`python -m edge.main --once`, etc.) still show
output. Individual modules just do `logging.getLogger(__name__)` and log
normally; only this module touches handlers.
"""

import logging
import logging.handlers
import os

from edge.config import EdgeConfig

_FILE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def configure_logging(config: EdgeConfig) -> None:
    """
    Configure the "edge" logger tree per config.storage.log_path and
    config.logging.level.

    Idempotent: clears any handlers already on the "edge" logger first, so
    calling this more than once (e.g. across tests) doesn't accumulate
    duplicate handlers / duplicated log lines.
    """
    logger = logging.getLogger("edge")
    logger.setLevel(config.logging.level.upper())
    logger.propagate = False
    logger.handlers.clear()

    log_path = config.storage.log_path
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
    )
    file_handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
    logger.addHandler(file_handler)

    # Message-only on the console -- matches the plain human-readable lines
    # edge/main.py used to print() directly, so --once/interactive output
    # (including the --once JSON summary) stays exactly as readable/parseable
    # as before; full timestamp/level/logger-name detail goes to the file.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
