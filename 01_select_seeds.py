# ==========================================================
# 01_select_seeds.py
# Select strong negative-ZFS seeds and retain CCDC provenance.
# ==========================================================

import pandas as pd
import os

mode = os.getenv("MODE", "crystal").lower()

if mode == "optimized":
    df = pd.read_csv("opt_D.csv")
    zfs_col = "opt_zfs"
else:
    df = pd.read_csv("GA.csv")
    zfs_col = "zfs"

print("MODE =", mode)
print("Database loaded:", "opt_D.csv" if mode == "optimized" else "GA.csv")

seed_df = df[df[zfs_col] <= -120].reset_index(drop=True)

# For optimized data, CCDC may not be present. Resolve later from GA.csv.
if "CCDC" not in seed_df.columns:
    seed_df["CCDC"] = ""

print("Seed complexes:", len(seed_df))

seed_df.to_csv("seed_complexes.csv", index=False)
print("[INFO] seed_complexes.csv saved with CCDC column")
