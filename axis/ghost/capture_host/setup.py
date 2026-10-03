#!/usr/bin/env python3
"""Set up ghost's capture environment: Playwright (pinned in requirements.txt), then its
Chromium - which Playwright skips when the shared ~/.cache/ms-playwright already has it.

    <capture_venv>/python setup.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements.txt")])
subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
print("capture environment ready")
