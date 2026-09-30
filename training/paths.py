"""Project paths; makes ../common (shared with the diagnosis service) importable."""
import os
import sys

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAINING_DIR)
COMMON_DIR = os.path.join(PROJECT_DIR, "common")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
