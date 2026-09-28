"""Compatibility shim for sip_train_v4_claude."""
import os
import sys

from sip_train_v4_claude import *
from sip_train_v4_claude import TrainingConfig, main

if __name__ == "__main__":
    import runpy
    _target = os.path.join(os.path.dirname(__file__), "sip_train_v4_claude.py")
    runpy.run_path(_target, run_name="__main__")
