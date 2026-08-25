"""DEPRECATED legacy E2E entrypoint. REFUSES TO RUN.

This file called the worker with the DEFAULT PRODUCTION NAMESPACE and mutated module
globals at import time. Both defects are fixed in the canonical path:
    runner  : _e2e_run.py   (explicit ns="e2e", explicit ExecPolicy)
    checker : _e2e_check.py
"""
import sys
sys.stderr.write("REFUSED: _e2e.py is deprecated. Use _e2e_run.py / _e2e_check.py "
                 "with S16_NS=e2e.\n")
sys.exit(2)
