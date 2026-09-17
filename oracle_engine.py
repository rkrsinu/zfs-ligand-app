# ==========================================================
# oracle_engine.py
#
# Loads the four GNN/scaler objects ONCE per GA worker process and
# reuses them for every generation.
# ==========================================================

import ast
import os
import pickle
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from complex_dataset import LigandCombinationDataset
from model import LigandGNN

NODE_FEATURE_DIM = 11
ED_CUTOFF = float(os.environ.get("ED_CUTOFF", 0.22))
ELITE_FRAC = float(os.environ.get("ELITE_FRAC", 0.10))


class OracleEngine:
    def __init__(self, mode="crystal", device=None):
        self.mode = str(mode).lower()
        if self.mode not in {"crystal", "optimized"}:
            raise ValueError("mode must be 'crystal' or 'optimized'")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.mode == "crystal":
            self.zfs_model_file = "zfs_gnn_crystal.pth"
            self.zfs_scaler_file = "zfs_scaler_crystal.pkl"
            self.ed_model_file = "ed_gnn_crystal.pth"
            self.ed_scaler_file = "ed_scaler_crystal.pkl"
        else:
            self.zfs_model_file = "zfs_gnn_opt.pth"
            self.zfs_scaler_file = "zfs_scaler_opt.pkl"
            self.ed_model_file = "ed_gnn_opt.pth"
            self.ed_scaler_file = "ed_scaler_opt.pkl"

        self.zfs_model, self.zfs_scaler = self._load_model(
            self.zfs_model_file, self.zfs_scaler_file
        )
        self.ed_model, self.ed_scaler = self._load_model(
            self.ed_model_file, self.ed_scaler_file
        )

        print(f"[ORACLE] MODE={self.mode}; DEVICE={self.device}")
        print("[ORACLE] GNN models loaded once and will be reused for all generations.")

    def _load_model(self, model_file, scaler_file):
        model = LigandGNN(node_feature_dim=NODE_FEATURE_DIM).to(self.device)
        state = torch.load(model_file, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        with open(scaler_file, "rb") as fh:
            scaler = pickle.load(fh)
        return model, scaler

    def predict_dataframe(self, df):
        if df.empty:
            out = df.copy()
            out["zfs_pred"] = []
            out["ed_pred"] = []
            return out

        ligand_lists = df["ligands"].astype(str).str.split(";").tolist()
        donor_lists = [[0] * 6 for _ in ligand_lists]
        da_lists = [["X"] * 6 for _ in ligand_lists]
        dummy_y = [0.0] * len(ligand_lists)

        dataset = LigandCombinationDataset(
            ligand_lists, donor_lists, da_lists, dummy_y
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        zfs_preds = []
        ed_preds = []

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device)
                z = self.zfs_model(batch).detach().cpu().numpy().reshape(-1, 1)
                e = self.ed_model(batch).detach().cpu().numpy().reshape(-1, 1)
                zfs_preds.extend(self.zfs_scaler.inverse_transform(z).flatten())
                ed_preds.extend(self.ed_scaler.inverse_transform(e).flatten())

        out = df.copy()
        out["zfs_pred"] = zfs_preds
        out["ed_pred"] = ed_preds
        return out

    @staticmethod
    def _ccdc_display(value):
        try:
            vals = ast.literal_eval(str(value))
            vals = [str(x) for x in vals if str(x).strip() and str(x).lower() != "nan"]
            return ";".join(vals)
        except Exception:
            return str(value)

    def screen(self, generated_file="generated_complexes.csv", target_zfs=-180.0):
        df = pd.read_csv(generated_file)
        print(f"[ORACLE] Generated complexes: {len(df)}")

        if df.empty:
            cols = list(df.columns) + ["zfs_pred", "ed_pred", "abs_err", "parent_CCDC_for_experiment"]
            pd.DataFrame(columns=cols).to_csv("elite_parents.csv", index=False)
            return pd.DataFrame(), pd.DataFrame()

        pred = self.predict_dataframe(df)

        before = len(pred)
        pred = pred[pred["ed_pred"] <= ED_CUTOFF].copy()
        print(f"[ORACLE] Passed E/D <= {ED_CUTOFF}: {len(pred)} / {before}")

        if pred.empty:
            cols = list(df.columns) + ["zfs_pred", "ed_pred", "abs_err", "parent_CCDC_for_experiment"]
            pd.DataFrame(columns=cols).to_csv("elite_parents.csv", index=False)
            pred.to_csv("oracle_screened_complexes.csv", index=False)
            return pd.DataFrame(), pred

        pred["abs_err"] = (pred["zfs_pred"] - float(target_zfs)).abs()
        pred.sort_values("abs_err", inplace=True)

        n_elite = max(1, int(len(pred) * ELITE_FRAC))
        elite = pred.head(n_elite).copy()
        elite["parent_CCDC_for_experiment"] = elite["parent_ccdcs"].apply(self._ccdc_display)

        elite.to_csv("elite_parents.csv", index=False)
        pred.to_csv("oracle_screened_complexes.csv", index=False)

        best = elite.iloc[0]
        print(f"[ORACLE] Elite={len(elite)}; best ZFS={best['zfs_pred']:.4f}; E/D={best['ed_pred']:.5f}")
        return elite, pred
