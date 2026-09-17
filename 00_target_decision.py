# ==========================================================
# 00_target_decision.py
# Direct database hit within ±10 cm⁻¹
# NUMERIC FILENAMES ONLY
#
# CCDC ADDITION ONLY: CCDC is merged into retrieved_solution.csv.
# The original hit-selection logic is unchanged.
# ==========================================================

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    sys.exit(2)

TARGET_ZFS = float(sys.argv[1])
MODE = os.environ.get("MODE", "X-ray")

if MODE == "X-ray":
    CSV_FILE = os.path.join(BASE_DIR, "GA.csv")
    ZFS_COL = "zfs"

elif MODE == "DFT":
    CSV_FILE = os.path.join(BASE_DIR, "opt_D.csv")
    ZFS_COL = "opt_zfs"

else:
    raise ValueError(f"Unknown MODE: {MODE}")

TOL = 10.0

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] DB = {CSV_FILE}")
print(f"[INFO] Target = {TARGET_ZFS}")

df = pd.read_csv(CSV_FILE)

df[ZFS_COL] = pd.to_numeric(df[ZFS_COL], errors="coerce")
df = df.dropna(subset=[ZFS_COL])

df["dist"] = (df[ZFS_COL] - TARGET_ZFS).abs()

hits = df[df["dist"] <= TOL].copy()

# =========================
# KEEP ONLY NUMERIC FILENAMES — UNCHANGED
# =========================
if "FileName" in hits.columns:
    hits = hits[
        hits["FileName"].astype(str).str.fullmatch(r"\d+")
    ]

# =========================
# CCDC metadata addition
# =========================
if "CCDC" not in hits.columns:
    lookup_path = os.path.join(BASE_DIR, "CCDC_lookup.csv")
    if os.path.exists(lookup_path) and "FileName" in hits.columns:
        cdf = pd.read_csv(lookup_path)
        cdf["FileName"] = cdf["FileName"].astype(str).str.strip()
        cdf = cdf.drop_duplicates("FileName")
        hits["_lookup_file"] = hits["FileName"].astype(str).str.strip()
        hits = hits.merge(cdf.rename(columns={"CCDC": "CCDC"}), left_on="_lookup_file", right_on="FileName", how="left", suffixes=("", "_ccdc"))
        if "FileName_ccdc" in hits.columns:
            hits.drop(columns=["FileName_ccdc"], inplace=True)
        hits.drop(columns=["_lookup_file"], inplace=True)
    else:
        hits["CCDC"] = ""

if len(hits) > 0:
    hits = hits.sort_values("dist").reset_index(drop=True)

    hits.to_csv(
        os.path.join(BASE_DIR, "retrieved_solution.csv"),
        index=False
    )

    print("🎯 DATABASE HIT")
    if "CCDC" in hits.columns:
        print("[INFO] CCDC:", ", ".join(hits["CCDC"].dropna().astype(str).unique()))
    sys.exit(0)

print("⚠️ NO DB HIT")
sys.exit(1)
