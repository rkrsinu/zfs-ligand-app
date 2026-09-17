# ==========================================================
# 04_build_complexes.py
# Memory-augmented complex construction
# OUTPUT: generated_complexes.csv
#
# CCDC ADDITION ONLY: provenance columns are added to each
# generated complex. Sampling/generation logic is unchanged.
# ==========================================================

import os
import random
import ast
import pandas as pd
from collections import Counter
import math

random.seed(42)

TARGET = 6
N_COMPLEXES = 5000
GEN = int(os.environ.get("GA_GEN", 0))

ALLOWED_PATTERNS = [
    (6,), (3,3), (4,1,1), (2,2,2),
    (1,2,3), (1,1,1,1,1,1), (5,1)
]

# ----------------------------------------------------------
# Load ligands
# ----------------------------------------------------------
df_real = pd.read_csv("ligand_donor_modes.csv")
df_mut  = pd.read_csv("mutated_ligands.csv")
df = pd.concat([df_real, df_mut], ignore_index=True)

MODE_MAP = df.groupby("smiles")["donors"].apply(list).to_dict()
ligands = list(MODE_MAP.keys())

# ----------------------------------------------------------
# CCDC provenance map (does not affect sampling)
# ----------------------------------------------------------
def norm(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def parse_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        v = ast.literal_eval(str(value))
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
    except Exception:
        pass
    return [x.strip() for x in str(value).split(";") if x.strip()]

CCDC_BY_FILE = {}
if os.path.exists("CCDC_lookup.csv"):
    cdf = pd.read_csv("CCDC_lookup.csv")
    for _, r in cdf.iterrows():
        f, c = norm(r.get("FileName")), norm(r.get("CCDC"))
        if f and c:
            CCDC_BY_FILE[f] = c

PROV_MAP = {}
if not df_mut.empty:
    for lig, g in df_mut.groupby("smiles", sort=False):
        r = g.iloc[0]
        PROV_MAP[lig] = {
            "parent_ligand": norm(r.get("parent_ligand")) or lig,
            "parent_ccdc": norm(r.get("parent_ccdc")),
            "parent_file_name": norm(r.get("parent_file_name")),
            "mutation": norm(r.get("mutation")) or "database_ligand",
        }

# Original reported ligand -> CCDC mapping from the CCDC lookup dataset.
if os.path.exists("GA.csv") and CCDC_BY_FILE:
    ga = pd.read_csv("GA.csv")
    for _, r in ga.iterrows():
        f = norm(r.get("FileName"))
        ccdc = CCDC_BY_FILE.get(f, "")
        if not ccdc:
            continue
        for i in range(1, 7):
            lig = norm(r.get(f"L{i}"))
            if not lig or lig.upper() == "X":
                continue
            PROV_MAP.setdefault(lig, {
                "parent_ligand": lig,
                "parent_ccdc": ccdc,
                "parent_file_name": f,
                "mutation": "database_ligand",
            })

for lig in ligands:
    PROV_MAP.setdefault(lig, {
        "parent_ligand": lig,
        "parent_ccdc": "",
        "parent_file_name": "",
        "mutation": "database_ligand",
    })

# ----------------------------------------------------------
# Pattern weights (soft memory) — UNCHANGED
# ----------------------------------------------------------
pattern_weights = Counter({p: 1.0 for p in ALLOWED_PATTERNS})

if os.path.exists("elite_parents.csv"):
    elite = pd.read_csv("elite_parents.csv").sort_values("zfs_pred")
    best = elite.iloc[0]
    try:
        best_pattern = tuple(sorted(eval(best["donor_list"])))
        pattern_weights[best_pattern] = min(
            pattern_weights[best_pattern] + 1.5, 4.0
        )
    except Exception:
        pass

# ----------------------------------------------------------
# Sampling — EXACTLY THE OLD LOGIC
# ----------------------------------------------------------
TEMP = 1.5
patterns = list(pattern_weights.keys())
weights  = [math.exp(pattern_weights[p] / TEMP) for p in patterns]

rows = []

while len(rows) < N_COMPLEXES:
    pattern = random.choices(patterns, weights)[0]
    used = set()
    chosen = []

    for d in pattern:
        cands = [l for l in ligands if l not in used and d in MODE_MAP[l]]
        if not cands:
            break
        lig = random.choice(cands)
        used.add(lig)
        chosen.append((lig, d))

    if sum(d for _, d in chosen) != TARGET:
        continue

    # CCDC/provenance is metadata only; it does not influence selection.
    parent_ligands = []
    parent_ccdcs = []
    parent_file_names = []
    mutations = []
    for lig, _ in chosen:
        p = PROV_MAP.get(lig, {})
        parent_ligands.append(p.get("parent_ligand", lig))
        parent_ccdcs.append(p.get("parent_ccdc", ""))
        parent_file_names.append(p.get("parent_file_name", ""))
        mutations.append(p.get("mutation", "database_ligand"))

    rows.append({
        "ligands": ";".join(l for l, _ in chosen),
        "donor_list": str([d for _, d in chosen]),
        "donor_sum": TARGET,
        "parent_ligands": str(parent_ligands),
        "parent_ccdcs": str(parent_ccdcs),
        "parent_file_names": str(parent_file_names),
        "mutations": str(mutations),
        "generation": GEN,
    })

pd.DataFrame(rows).to_csv("generated_complexes.csv", index=False)
print("[INFO] Generated complexes:", len(rows))
