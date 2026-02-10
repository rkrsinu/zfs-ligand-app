# ============================================================
# graph_features.py
#
# SMILES -> molecular graph with 11-dim node features
# NaN-safe and PyTorch-Geometric compatible
# ============================================================

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PAULING_EN = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44,
    "P": 2.19, "S": 2.58, "SE": 2.55
}

def atom_en(sym):
    return float(PAULING_EN.get(str(sym).upper(), 0.0))

def _safe_float(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0

def smiles_to_graph(smiles, donor_symbol="X"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    feats = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()

        g = 0.0
        if atom.HasProp("_GasteigerCharge"):
            g = _safe_float(atom.GetProp("_GasteigerCharge"))

        feat = [
            _safe_float(atom.GetAtomicNum()),
            _safe_float(atom.GetDegree()),
            _safe_float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            _safe_float(atom.GetFormalCharge()),
            _safe_float(atom.GetImplicitValence()),
            1.0,
            atom_en(sym),
            1.0 if sym.upper() == str(donor_symbol).upper() else 0.0,
            atom_en(sym) if sym.upper() == str(donor_symbol).upper() else 0.0,
            g
        ]

        feats.append([_safe_float(v) for v in feat])

    x = torch.tensor(feats, dtype=torch.float32)

    ei0, ei1 = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei0.extend([i, j])
        ei1.extend([j, i])

    if len(ei0) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([ei0, ei1], dtype=torch.long)

    return x, edge_index
