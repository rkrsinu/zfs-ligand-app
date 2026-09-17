# ==========================================================
# 06_run_until_target.py
# Command-line entry point for the same persistent worker used
# by the Streamlit application.
# ==========================================================

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 3:
    print("Usage: python 06_run_until_target.py <TARGET_ZFS> <MODE> [MAX_GEN]")
    print("MODE = crystal | optimized")
    sys.exit(2)

TARGET = float(sys.argv[1])
MODE = sys.argv[2].lower()
MAX_GEN = int(sys.argv[3]) if len(sys.argv) >= 4 else int(os.environ.get("MAX_GEN", "500"))

if MODE not in {"crystal", "optimized"}:
    print("MODE must be crystal or optimized")
    sys.exit(2)

cmd = [
    sys.executable,
    os.path.join(BASE_DIR, "ga_worker.py"),
    "--target", str(TARGET),
    "--mode", MODE,
    "--max-generations", str(MAX_GEN),
    "--drive-sync-every", os.environ.get("DRIVE_SYNC_EVERY", "5"),
    "--n-complexes", os.environ.get("N_COMPLEXES", "5000"),
]

raise SystemExit(subprocess.call(cmd, cwd=BASE_DIR))
