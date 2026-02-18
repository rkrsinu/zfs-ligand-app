# ==========================================================
# 00_target_decision.py
# MODE-aware retrieval gate (±10 direct hit)
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

TOL = 10.0   # ✅ YOUR REQUIRED WINDOW

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] Using {CSV_FILE}")
print(f"[INFO] Target = {TARGET_ZFS}  Tol = ±{TOL}")

# ---------------- LOAD DATABASE ----------------
df = pd.read_csv(CSV_FILE)

# Ensure numeric (VERY IMPORTANT for Streamlit runtime)
df[ZFS_COL] = pd.to_numeric(df[ZFS_COL], errors="coerce")

# Compute distance
df["dist"] = (df[ZFS_COL] - TARGET_ZFS).abs()

# Find ALL hits within tolerance
hits = df[df["dist"] <= TOL].copy()

# ----------------------------------------------------------
# DIRECT DATABASE RETURN
# ----------------------------------------------------------
if len(hits) > 0:

    hits = hits.sort_values("dist")

    hits.to_csv("retrieved_solution.csv", index=False)

    print("🎯 Retrieved from database (within tolerance)")
    print(hits[[ZFS_COL, "dist"]].head())

    sys.exit(0)

print("⚠️ No database match — switching to GA")
sys.exit(1)
