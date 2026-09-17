import pandas as pd

df = pd.read_csv("opt_D.csv")

ligand_modes = {}

for _, row in df.iterrows():
    for i in range(1, 7):
        lig = row.get(f"L{i}")
        d   = row.get(f"D{i}")

        if not isinstance(lig, str):
            continue
        if lig == "X":
            continue
        if pd.isna(d):
            continue

        ligand_modes.setdefault(lig, set()).add(int(d))

rows = []
for lig, modes in ligand_modes.items():
    for m in modes:
        rows.append({"smiles": lig, "donors": m})

out = pd.DataFrame(rows)
out.to_csv("ligand_donor_modes.csv", index=False)

print("[INFO] ligand_donor_modes.csv created")
print(out.groupby("donors").size())
