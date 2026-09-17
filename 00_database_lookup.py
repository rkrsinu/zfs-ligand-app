# ==========================================================
# 00_database_lookup.py
# MODE-aware database lookup
# ==========================================================

import os
import pandas as pd

MODE = os.environ.get("MODE", "crystal")

if MODE == "crystal":
    CSV_FILE = "GA.csv"
    ZFS_COL = "zfs"
else:
    CSV_FILE = "opt_D.csv"
    ZFS_COL = "opt_zfs"

df = pd.read_csv(CSV_FILE)
df.to_csv("working_database.csv", index=False)

print(f"[INFO] Loaded database: {CSV_FILE}")
