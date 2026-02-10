# ============================================================
# complex_dataset.py
#
# Robust dataset for octahedral Co complexes
# Guarantees:
#   - >=1 ligand graph per sample
#   - finite ZFS and E/D
# ============================================================

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np

from graph_features import smiles_to_graph

class ComplexDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # Normalize columns
        for i in range(1, 7):
            df[f"L{i}"] = (
                df.get(f"L{i}", "X")
                .astype(str)
                .str.strip()
                .replace({"": "X", "nan": "X", "None": "X"})
            )
            df[f"DA{i}"] = (
                df.get(f"DA{i}", "X")
                .astype(str)
                .str.strip()
                .replace({"": "X", "nan": "X", "None": "X"})
            )
            df[f"D{i}"] = (
                pd.to_numeric(df.get(f"D{i}", 0), errors="coerce")
                .fillna(0)
                .astype(int)
            )

        df["zfs"] = pd.to_numeric(df["zfs"], errors="coerce").fillna(0.0)
        df["E/D"] = pd.to_numeric(df["E/D"], errors="coerce").fillna(0.0)

        # Pre-filter rows with at least one valid ligand
        valid_rows = []
        for _, row in df.iterrows():
            has_lig = any(str(row[f"L{i}"]).upper() != "X" for i in range(1, 7))
            if has_lig:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("No valid complexes with ligands found!")

        print(f"[INFO] Loaded {len(self.df)} valid complexes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        graphs = []

        for i in range(1, 7):
            smi = row[f"L{i}"]
            da  = row[f"DA{i}"]

            if smi.upper() == "X":
                continue

            g = smiles_to_graph(smi, da)
            if g is None:
                continue

            x, ei = g
            graphs.append(Data(x=x, edge_index=ei))

        # SAFETY: ensure at least one graph
        if len(graphs) == 0:
            raise RuntimeError(f"Row {idx} has no valid ligand graphs")

        cond = torch.tensor(
            [float(row["zfs"]), float(row["E/D"])],
            dtype=torch.float32
        )

        return graphs, cond
