# ==========================================================
# 00_target_decision.py
# Direct database hit within ±10 cm⁻¹
# ==========================================================

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    sys.exit(2)

TARGET_ZFS = float(sys.argv[1])
MODE = os.environ.get("MODE", "crystal")

if MODE == "crystal":
    CSV_FILE = os.path.join(BASE_DIR, "GA.csv")
    ZFS_COL = "zfs"
else:
    CSV_FILE = os.path.join(BASE_DIR, "opt_D.csv")
    ZFS_COL = "opt_zfs"

TOL = 10.0

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] DB = {CSV_FILE}")
print(f"[INFO] Target = {TARGET_ZFS}")

df = pd.read_csv(CSV_FILE)

df[ZFS_COL] = pd.to_numeric(df[ZFS_COL], errors="coerce")
df = df.dropna(subset=[ZFS_COL])

df["dist"] = (df[ZFS_COL] - TARGET_ZFS).abs()

hits = df[df["dist"] <= TOL].copy()

if len(hits) > 0:
    hits = hits.sort_values("dist")
    hits.to_csv(os.path.join(BASE_DIR, "retrieved_solution.csv"), index=False)

    print("🎯 DATABASE HIT")
    sys.exit(0)

print("⚠️ NO DB HIT")
sys.exit(1)
