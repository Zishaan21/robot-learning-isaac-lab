"""Runs an Isaac Lab script (train.py / play.py) with the lesson-07 task registered.

Same idea as ..\\isaaclab_run.py (pre-load h5py), plus: put this folder on the Python path and
register "Lesson-PickDrop-Franka-v0" before the script looks the task up.

Usage (from Learning\\Robotics):
    run.bat lesson_07_pick_drop\\run_task.py <Isaac Lab script> --task Lesson-PickDrop-Franka-v0 [args...]
"""
import os
import runpy
import sys

import h5py  # noqa: F401  (must happen before Isaac Sim starts)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pick_drop_task  # noqa: E402,F401  (registers the task ids)

script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))  # train.py / play.py import cli_args.py from their folder
runpy.run_path(script, run_name="__main__")
