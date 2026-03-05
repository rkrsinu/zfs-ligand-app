# ==========================================================
# 05_oracle_screen.py
# Oracle screening (CRYSTAL + OPTIMIZED)
# NO retraining
# NO dimension guessing
# ==========================================================

import os
import torch
import pickle
import pandas as pd
from torch_geometric.loader import DataLoader

from ligand_dataset import LigandCombinationDataset
from model import LigandGNN

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔴 THIS IS THE KEY LINE
NODE_FEATURE_DIM = 11   # MUST match training-time features

MODE = os.environ.get("MODE", "optimized").lower()
TARGET_ZFS = float(os.environ.get("TARGET_ZFS", -180.0))

ED_CUTOFF = 0.22
ELITE_FRAC = 0.10

# ----------------------------------------------------------
# Model & scaler selection
# ----------------------------------------------------------

if MODE == "crystal":
    print("[INFO] MODE = crystal")
    ZFS_MODEL = "zfs_gnn_crystal.pth"
    ZFS_SCALER = "zfs_scaler_crystal.pkl"
    ED_MODEL = "ed_gnn_crystal.pth"
    ED_SCALER = "ed_scaler_crystal.pkl"
else:
    print("[INFO] MODE = optimized")
    ZFS_MODEL = "zfs_gnn_opt.pth"
    ZFS_SCALER = "zfs_scaler_opt.pkl"
    ED_MODEL = "ed_gnn_opt.pth"
    ED_SCALER = "ed_scaler_opt.pkl"

# ----------------------------------------------------------
# Load generated complexes
# ----------------------------------------------------------

df = pd.read_csv("generated_complexes.csv")
print("[INFO] Generated complexes:", len(df))

ligand_lists = df["ligands"].astype(str).str.split(";").tolist()

# Dummy lists (oracle-only inference)
donor_lists = [[0]*6] * len(ligand_lists)
da_lists = [["X"]*6] * len(ligand_lists)
dummy_y = [0.0] * len(ligand_lists)

# ----------------------------------------------------------
# Dataset & loader
# ----------------------------------------------------------

dataset = LigandCombinationDataset(
    ligand_lists,
    donor_lists,
    da_lists,
    dummy_y
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ----------------------------------------------------------
# Load ZFS model
# ----------------------------------------------------------

zfs_model = LigandGNN(node_feature_dim=NODE_FEATURE_DIM).to(DEVICE)
zfs_model.load_state_dict(torch.load(ZFS_MODEL, map_location=DEVICE))
zfs_model.eval()

with open(ZFS_SCALER, "rb") as f:
    zfs_scaler = pickle.load(f)

# ----------------------------------------------------------
# Load E/D model
# ----------------------------------------------------------

ed_model = LigandGNN(node_feature_dim=NODE_FEATURE_DIM).to(DEVICE)
ed_model.load_state_dict(torch.load(ED_MODEL, map_location=DEVICE))
ed_model.eval()

with open(ED_SCALER, "rb") as f:
    ed_scaler = pickle.load(f)

# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------

zfs_preds = []
ed_preds = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)

        z = zfs_model(batch).cpu().numpy().reshape(-1, 1)
        e = ed_model(batch).cpu().numpy().reshape(-1, 1)

        zfs_preds.extend(zfs_scaler.inverse_transform(z).flatten())
        ed_preds.extend(ed_scaler.inverse_transform(e).flatten())

df["zfs_pred"] = zfs_preds
df["ed_pred"] = ed_preds

# ----------------------------------------------------------
# Hard constraint: E/D cutoff
# ----------------------------------------------------------

df = df[df["ed_pred"] <= ED_CUTOFF].copy()
print(f"[INFO] Passed E/D filter (<= {ED_CUTOFF}): {len(df)}")

# ----------------------------------------------------------
# Rank by target ZFS
# ----------------------------------------------------------

df["abs_err"] = (df["zfs_pred"] - TARGET_ZFS).abs()
df.sort_values("abs_err", inplace=True)

# ----------------------------------------------------------
# Elite selection
# ----------------------------------------------------------

n_elite = max(1, int(len(df) * ELITE_FRAC))
elite = df.head(n_elite)

elite.to_csv("elite_parents.csv", index=False)

print("[INFO] Elite saved:", len(elite))
print("[INFO] Best predicted ZFS:", elite.iloc[0]["zfs_pred"])
