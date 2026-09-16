# ==========================================================
# 00_target_decision.py
# Direct database hit within ±10 cm⁻¹
# Updated to retain/display CCDC for experimental lookup.
# ==========================================================

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    sys.exit(2)

TARGET_ZFS = float(sys.argv[1])
MODE = os.environ.get("MODE", "crystal").lower()

if MODE in ("crystal", "x-ray", "xray"):
    CSV_FILE = os.path.join(BASE_DIR, "GA.csv")
    ZFS_COL = "zfs"
elif MODE in ("optimized", "dft"):
    CSV_FILE = os.path.join(BASE_DIR, "opt_D.csv")
    ZFS_COL = "opt_zfs"
else:
    raise ValueError(f"Unknown MODE: {MODE}")

TOL = float(os.environ.get("DB_TOL", 10.0))

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] DB = {CSV_FILE}")
print(f"[INFO] Target = {TARGET_ZFS}")

df = pd.read_csv(CSV_FILE)
df[ZFS_COL] = pd.to_numeric(df[ZFS_COL], errors="coerce")
df = df.dropna(subset=[ZFS_COL]).copy()
df["dist"] = (df[ZFS_COL] - TARGET_ZFS).abs()
hits = df[df["dist"] <= TOL].copy()

if "FileName" in hits.columns:
    hits = hits[hits["FileName"].astype(str).str.fullmatch(r"\d+")]

if len(hits) > 0:
    hits = hits.sort_values("dist").reset_index(drop=True)
    hits.to_csv(os.path.join(BASE_DIR, "retrieved_solution.csv"), index=False)

    print("🎯 DATABASE HIT")
    cols = [c for c in ["FileName", "CCDC", "L1", "L2", "L3", "L4", "L5", "L6", ZFS_COL, "E/D", "opt_E/D"] if c in hits.columns]
    print(hits[cols].head(10).to_string(index=False))
    if "CCDC" in hits.columns:
        print("[INFO] CCDC number(s) shown above can be used to locate the experimental synthesis.")
    sys.exit(0)

print("⚠️ NO DB HIT")
sys.exit(1)
