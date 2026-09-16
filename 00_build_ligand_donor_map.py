# ==========================================================
# 00_build_ligand_donor_map.py
# Build ligand -> donor-count map from the ACTIVE database.
#
# crystal   -> GA.csv (updated GA dataset with CCDC)
# optimized -> opt_D.csv
# ==========================================================

import os
import pandas as pd

MODE = os.environ.get("MODE", "crystal").lower()

if MODE == "crystal":
    csv_file = "GA.csv"
elif MODE == "optimized":
    csv_file = "opt_D.csv"
else:
    raise ValueError(f"Unknown MODE: {MODE}")

if not os.path.exists(csv_file):
    raise FileNotFoundError(csv_file)

df = pd.read_csv(csv_file)

ligand_modes = {}

for _, row in df.iterrows():
    for i in range(1, 7):
        lig = row.get(f"L{i}")
        d = row.get(f"D{i}")

        if not isinstance(lig, str):
            continue
        lig = lig.strip()
        if not lig or lig.upper() == "X":
            continue
        if pd.isna(d):
            continue

        try:
            donor = int(float(d))
        except Exception:
            continue

        ligand_modes.setdefault(lig, set()).add(donor)

rows = []
for lig, modes in sorted(ligand_modes.items()):
    for m in sorted(modes):
        rows.append({"smiles": lig, "donors": m})

out = pd.DataFrame(rows)
out.to_csv("ligand_donor_modes.csv", index=False)

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] Source database = {csv_file}")
print("[INFO] ligand_donor_modes.csv created")
print(f"[INFO] Unique ligands: {out['smiles'].nunique() if not out.empty else 0}")
print(out.groupby("donors").size() if not out.empty else "No donor modes found")
