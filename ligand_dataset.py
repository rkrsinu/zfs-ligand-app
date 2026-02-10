# ligand_dataset.py
"""
LigandCombinationDataset - updated to include donor atom features and sanitize non-finite values.
Each node has 11 features:
 [Z, degree, total_Hs, aromatic, formal_charge, implicit_valence, bias,
  atom_en, is_donor, donor_en, gasteiger_charge]
Fallback ligand-node returns the same 11 dims.
"""
import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

PAULING_EN = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "CL": 3.16, "BR": 2.96, "I": 2.66,
    "B": 2.04, "SI": 1.90, "SE": 2.55, "ZN": 1.65, "FE": 1.83
}

def atom_en(sym: str) -> float:
    if sym is None:
        return 0.0
    return float(PAULING_EN.get(str(sym).strip().upper(), 0.0))

def build_mol_graph_from_smiles_with_donor(smiles: str, donor_symbol: str = None):
    if not RDKit_AVAILABLE:
        raise RuntimeError("RDKit not available")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.tensor([[0.0]*11], dtype=torch.float32), torch.zeros((2,0), dtype=torch.long)

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    feats = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        z = float(atom.GetAtomicNum())
        deg = float(atom.GetDegree())
        tot_h = float(atom.GetTotalNumHs())
        arom = float(atom.GetIsAromatic())
        fcharge = float(atom.GetFormalCharge())
        impval = float(atom.GetImplicitValence())
        bias = 1.0

        atom_en_val = atom_en(sym)
        is_donor = 0.0
        donor_en = 0.0
        if donor_symbol and str(donor_symbol).strip().upper() not in ("", "X", "NAN", "NONE"):
            if sym.upper() == str(donor_symbol).strip().upper():
                is_donor = 1.0
                donor_en = atom_en_val

        g_charge = 0.0
        try:
            if atom.HasProp("_GasteigerCharge"):
                raw = atom.GetProp("_GasteigerCharge")
                g_charge = float(str(raw))
                if not np.isfinite(g_charge):
                    g_charge = 0.0
        except Exception:
            g_charge = 0.0

        feat = [z, deg, tot_h, arom, fcharge, impval, bias,
                atom_en_val, is_donor, donor_en, g_charge]
        feat = [0.0 if (not np.isfinite(float(v))) else float(v) for v in feat]
        feats.append(feat)

    x = torch.tensor(feats, dtype=torch.float32)

    ei0, ei1 = [], []
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx(); c = b.GetEndAtomIdx()
        ei0.extend([a, c]); ei1.extend([c, a])
    if len(ei0) == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor([ei0, ei1], dtype=torch.long)

    return x, edge_index

def build_fallback_ligand_node_feature(smiles: str, donor_symbol: str):
    da = str(donor_symbol).strip().upper() if donor_symbol is not None else ""
    donor_en = float(PAULING_EN.get(da, 0.0)) if da not in ("", "X", "NAN", "NONE") else 0.0
    is_flag = 0.0 if da in ("", "X", "NAN", "NONE") else 1.0
    feat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, donor_en, is_flag, donor_en, 0.0]
    feat = [0.0 if (not np.isfinite(float(v))) else float(v) for v in feat]
    return torch.tensor(feat, dtype=torch.float32)

class LigandCombinationDataset(InMemoryDataset):
    def __init__(self, smiles_lists, donor_lists, da_lists, y, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        data_list = []
        n = len(y)
        for i in range(n):
            s_row = smiles_lists[i] if smiles_lists is not None else ["X"]*6
            d_row = donor_lists[i] if donor_lists is not None else [0]*6
            da_row = da_lists[i] if da_lists is not None else ["X"]*6
            data = self._build_row_graph(s_row, d_row, da_row, y[i], row_index=i)
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)

    def _build_row_graph(self, smiles_row, donor_row, da_row, y_value, row_index=0):
        node_feats = []
        edge_list = []
        offset = 0

        for i in range(6):
            smi = str(smiles_row[i]).strip() if i < len(smiles_row) else "X"
            da = str(da_row[i]).strip() if i < len(da_row) else "X"

            if smi.upper() in ("", "X", "NAN", "NONE"):
                xi = build_fallback_ligand_node_feature(smi, da).unsqueeze(0)
                ei = torch.zeros((2, 0), dtype=torch.long)
            else:
                if RDKit_AVAILABLE:
                    try:
                        xi, ei = build_mol_graph_from_smiles_with_donor(smi, da)
                    except Exception:
                        xi = build_fallback_ligand_node_feature(smi, da).unsqueeze(0)
                        ei = torch.zeros((2, 0), dtype=torch.long)
                else:
                    xi = build_fallback_ligand_node_feature(smi, da).unsqueeze(0)
                    ei = torch.zeros((2, 0), dtype=torch.long)

            if xi.dim() == 1:
                xi = xi.unsqueeze(0)
            node_feats.append(xi)

            if ei.numel() > 0:
                if ei.dim() != 2 or ei.size(0) != 2:
                    raise RuntimeError(f"Unexpected edge_index shape for ligand {i} in row {row_index}: {ei.shape}")
                ei_off = ei + offset
                edge_list.append(ei_off)
            offset += xi.size(0)

        feat_dims = [int(tensor.size(1)) for tensor in node_feats]
        if len(set(feat_dims)) != 1:
            raise RuntimeError(
                f"Node feature dim mismatch in row {row_index}: per-ligand feature dims = {feat_dims}. "
                "Make sure RDKit and fallback paths produce the same feature length (11)."
            )

        x = torch.cat(node_feats, dim=0)

        if len(edge_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.cat(edge_list, dim=1)

        data = Data(x=x, edge_index=edge_index)
        data.y = torch.tensor([float(y_value)], dtype=torch.float32)
        return data
