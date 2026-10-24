"""Runs an Isaac Lab script after pre-loading h5py.

Why: Isaac Sim's RTX-sensor extension ships its own hdf5.dll. In GUI runs it loads first, and then
`import h5py` (done by Isaac Lab at startup) fails with
"DLL load failed while importing _errors" / Windows error 0xc0000139.
Importing h5py before Isaac Sim starts makes h5py's own hdf5.dll the one in memory.

Usage (from Learning\\Robotics):
    run.bat isaaclab_run.py <path to Isaac Lab script> [script args...]
"""
import os
import runpy
import sys

import h5py  # noqa: F401  (must happen before Isaac Sim starts)

script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]  # the script sees its own path as argv[0], like a normal run
sys.path.insert(0, os.path.dirname(script))  # e.g. play.py imports cli_args.py from its own folder
runpy.run_path(script, run_name="__main__")
