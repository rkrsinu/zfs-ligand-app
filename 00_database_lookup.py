# ==========================================================
# 00_database_lookup.py
# MODE-aware database lookup
# ==========================================================

import os
import pandas as pd

MODE = os.environ.get("MODE", "crystal").lower()

if MODE == "crystal":
    CSV_FILE = "GA.csv"
    ZFS_COL = "zfs"
elif MODE == "optimized":
    CSV_FILE = "opt_D.csv"
    ZFS_COL = "opt_zfs"
else:
    raise ValueError(f"Unknown MODE: {MODE}")

df = pd.read_csv(CSV_FILE)
df.to_csv("working_database.csv", index=False)

print(f"[INFO] Loaded database: {CSV_FILE}")
print(f"[INFO] ZFS column: {ZFS_COL}")
if "CCDC" in df.columns:
    print("[INFO] CCDC column available for experimental lookup.")
