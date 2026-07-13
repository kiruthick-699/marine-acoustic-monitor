"""
Regression test for edge/main.py's logger name.

`python -m edge.main` sets that module's __name__ to "__main__" (not
"edge.main"), so `logging.getLogger(__name__)` there would silently detach
from the "edge" logger tree edge/logging_setup.py configures -- no handlers,
default root level -- dropping every log line the entrypoint itself emits
(calibration/scheduler logs still worked, since those modules are always
imported, never run as __main__, masking the bug when only spot-checking
their output).
"""

import edge.main


def test_main_module_logger_name_is_explicit_not_dunder_name():
    assert edge.main.logger.name == "edge.main"
