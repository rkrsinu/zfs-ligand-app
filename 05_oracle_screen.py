# ==========================================================
# 05_oracle_screen.py
# Stand-alone compatibility wrapper.
#
# The Streamlit/long-running GA now uses oracle_engine.py so that
# the GNN models are loaded only once per worker process.
# ==========================================================

import os
from oracle_engine import OracleEngine

MODE = os.environ.get("MODE", "optimized").lower()
TARGET_ZFS = float(os.environ.get("TARGET_ZFS", -180.0))

engine = OracleEngine(MODE)
engine.screen("generated_complexes.csv", TARGET_ZFS)
