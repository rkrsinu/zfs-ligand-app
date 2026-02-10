# ==========================================================
# test_model_load.py
# Windows-safe test for checkpoint loading
# ==========================================================

import torch
from ligand_gnn_model import LigandGNN

print("Loading model...")

model = LigandGNN()
state = torch.load("zfs_gnn_crystal.pth", map_location="cpu")

model.load_state_dict(state)

print("✅ ZFS model loaded perfectly")
