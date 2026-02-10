import pandas as pd

df = pd.read_csv("seed_complexes.csv")

ligands = set()
for i in range(1, 7):
    ligands.update(df[f"L{i}"].dropna().astype(str))

ligands = sorted(ligands)

pd.DataFrame({"smiles": ligands}).to_csv(
    "seed_ligands.csv", index=False
)

print("Seed ligands:", len(ligands))
