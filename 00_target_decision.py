# ==========================================================
# 00_target_decision.py
# MODE-aware retrieval gate
# ==========================================================

import os
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python 00_target_decision.py <TARGET_ZFS>")
    sys.exit(2)

TARGET_ZFS = float(sys.argv[1])
MODE = os.environ.get("MODE", "crystal")

# ---------------- MODE SWITCH ----------------
if MODE == "crystal":
    CSV_FILE = "GA.csv"
    ZFS_COL = "zfs"
else:
    CSV_FILE = "opt_D.csv"
    ZFS_COL = "opt_zfs"

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] Using {CSV_FILE}")

# ---------------- LOAD DATABASE ----------------
df = pd.read_csv(CSV_FILE)

df["dist"] = abs(df[ZFS_COL] - TARGET_ZFS)

best = df.sort_values("dist").iloc[0]

# tolerance (cm-1)
TOL = 5.0

if best["dist"] <= TOL:
    best.to_frame().T.to_csv("retrieved_solution.csv", index=False)
    print("🎯 Retrieved from database")
    sys.exit(0)

print("⚠️ No database match — switching to GA")
sys.exit(1)
