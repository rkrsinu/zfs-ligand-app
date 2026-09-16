# ==========================================================
# 02_extract_seed_ligands.py
# Extract seed ligands AND retain their exact parent CCDC.
# ==========================================================

import pandas as pd

seed_df = pd.read_csv("seed_complexes.csv")

rows = []
for _, row in seed_df.iterrows():
    ccdc = row.get("CCDC", "")
    file_name = row.get("FileName", row.get("File Name", ""))

    ccdc = "" if pd.isna(ccdc) else str(ccdc).strip()
    file_name = "" if pd.isna(file_name) else str(file_name).strip()

    for i in range(1, 7):
        lig = row.get(f"L{i}")
        donor = row.get(f"D{i}")

        if not isinstance(lig, str) or lig.strip().upper() == "X":
            continue
        if pd.isna(donor):
            continue

        rows.append({
            "smiles": lig.strip(),
            "donors": int(donor),
            "parent_ligand": lig.strip(),
            "parent_ccdc": ccdc,
            "parent_file_name": file_name,
            "generation": 0,
            "mutation": "database_seed",
        })

out = pd.DataFrame(rows).drop_duplicates()
out.to_csv("seed_ligands.csv", index=False)

print("Seed ligands:", out["smiles"].nunique())
print("Seed ligand records:", len(out))
print("[INFO] Parent CCDC retained in seed_ligands.csv")
