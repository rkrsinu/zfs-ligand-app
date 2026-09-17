# ==========================================================
# 05_oracle_screen.py
# Oracle screening (CRYSTAL + OPTIMIZED)
# NO retraining
# NO dimension guessing
# Preserves experimental provenance/CCDC lineage.
# ==========================================================

import os
import ast
import torch
import pickle
import pandas as pd
from torch_geometric.loader import DataLoader

from ligand_dataset import LigandCombinationDataset
from model import LigandGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NODE_FEATURE_DIM = 11
MODE = os.environ.get("MODE", "optimized").lower()
TARGET_ZFS = float(os.environ.get("TARGET_ZFS", -180.0))
ED_CUTOFF = float(os.environ.get("ED_CUTOFF", 0.22))
ELITE_FRAC = float(os.environ.get("ELITE_FRAC", 0.10))

if MODE == "crystal":
    print("[INFO] MODE = crystal")
    ZFS_MODEL = "zfs_gnn_crystal.pth"
    ZFS_SCALER = "zfs_scaler_crystal.pkl"
    ED_MODEL = "ed_gnn_crystal.pth"
    ED_SCALER = "ed_scaler_crystal.pkl"
elif MODE == "optimized":
    print("[INFO] MODE = optimized")
    ZFS_MODEL = "zfs_gnn_opt.pth"
    ZFS_SCALER = "zfs_scaler_opt.pkl"
    ED_MODEL = "ed_gnn_opt.pth"
    ED_SCALER = "ed_scaler_opt.pkl"
else:
    raise ValueError(f"Unknown MODE: {MODE}. Use crystal or optimized.")

# ----------------------------------------------------------
# Load generated complexes
# ----------------------------------------------------------
df = pd.read_csv("generated_complexes.csv")
print("[INFO] Generated complexes:", len(df))

ligand_lists = df["ligands"].astype(str).str.split(";").tolist()
donor_lists = [[0] * 6 for _ in ligand_lists]
da_lists = [["X"] * 6 for _ in ligand_lists]
dummy_y = [0.0] * len(ligand_lists)

dataset = LigandCombinationDataset(ligand_lists, donor_lists, da_lists, dummy_y)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ----------------------------------------------------------
# Model helper
# ----------------------------------------------------------
def load_model(model_file, scaler_file):
    model = LigandGNN(node_feature_dim=NODE_FEATURE_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

zfs_model, zfs_scaler = load_model(ZFS_MODEL, ZFS_SCALER)
ed_model, ed_scaler = load_model(ED_MODEL, ED_SCALER)

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
# Hard E/D constraint
# ----------------------------------------------------------
before = len(df)
df = df[df["ed_pred"] <= ED_CUTOFF].copy()
print(f"[INFO] Passed E/D filter (<= {ED_CUTOFF}): {len(df)} / {before}")

if df.empty:
    print("[WARNING] No complexes passed the E/D cutoff.")
    pd.DataFrame(columns=list(pd.read_csv("generated_complexes.csv", nrows=0).columns) + ["zfs_pred", "ed_pred", "abs_err"]).to_csv("elite_parents.csv", index=False)
    raise SystemExit(0)

# ----------------------------------------------------------
# Rank by target ZFS
# ----------------------------------------------------------
df["abs_err"] = (df["zfs_pred"] - TARGET_ZFS).abs()
df.sort_values("abs_err", inplace=True)

# ----------------------------------------------------------
# Elite selection
# ----------------------------------------------------------
n_elite = max(1, int(len(df) * ELITE_FRAC))
elite = df.head(n_elite).copy()

# Make CCDC easy to read in the final output.
def ccdc_display(value):
    try:
        vals = ast.literal_eval(str(value))
        vals = [str(x) for x in vals if str(x).strip() and str(x).lower() != "nan"]
        return ";".join(vals)
    except Exception:
        return str(value)

elite["parent_CCDC_for_experiment"] = elite["parent_ccdcs"].apply(ccdc_display)

elite.to_csv("elite_parents.csv", index=False)
df.to_csv("oracle_screened_complexes.csv", index=False)

best = elite.iloc[0]
print("[INFO] Elite saved:", len(elite))
print(f"[INFO] Best predicted ZFS: {best['zfs_pred']:.4f}")
print(f"[INFO] Best E/D: {best['ed_pred']:.4f}")
