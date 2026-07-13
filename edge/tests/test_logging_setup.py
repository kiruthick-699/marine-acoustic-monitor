"""
Tests for edge/logging_setup.py -- the "edge" logger tree's rotating file
handler + console handler wiring.
"""

import logging
import logging.handlers
import os

from edge.config import EdgeConfig
from edge.logging_setup import configure_logging


def _make_config(tmp_path, level: str = "INFO") -> EdgeConfig:
    config = EdgeConfig()
    config.storage.log_path = str(tmp_path / "edge.log")
    config.logging.level = level
    return config


def test_configure_logging_creates_log_file_and_writes_records(tmp_path):
    config = _make_config(tmp_path)
    configure_logging(config)

    logger = logging.getLogger("edge.some_module")
    logger.info("hello from a test")
    for handler in logging.getLogger("edge").handlers:
        handler.flush()

    assert os.path.exists(config.storage.log_path)
    with open(config.storage.log_path) as f:
        contents = f.read()
    assert "hello from a test" in contents


def test_configure_logging_attaches_file_and_console_handlers(tmp_path):
    config = _make_config(tmp_path)
    configure_logging(config)

    logger = logging.getLogger("edge")
    handler_types = {type(h) for h in logger.handlers}
    assert logging.handlers.RotatingFileHandler in handler_types
    assert logging.StreamHandler in handler_types


def test_configure_logging_is_idempotent_no_duplicate_handlers(tmp_path):
    config = _make_config(tmp_path)
    configure_logging(config)
    configure_logging(config)

    logger = logging.getLogger("edge")
    assert len(logger.handlers) == 2


def test_configure_logging_applies_configured_level(tmp_path):
    config = _make_config(tmp_path, level="DEBUG")
    configure_logging(config)
    assert logging.getLogger("edge").getEffectiveLevel() == logging.DEBUG

    config = _make_config(tmp_path, level="WARNING")
    configure_logging(config)
    assert logging.getLogger("edge").getEffectiveLevel() == logging.WARNING


def test_configure_logging_respects_level_for_child_loggers(tmp_path):
    config = _make_config(tmp_path, level="WARNING")
    configure_logging(config)

    child_logger = logging.getLogger("edge.calibration")
    assert child_logger.getEffectiveLevel() == logging.WARNING
    assert child_logger.isEnabledFor(logging.INFO) is False
    assert child_logger.isEnabledFor(logging.WARNING) is True
