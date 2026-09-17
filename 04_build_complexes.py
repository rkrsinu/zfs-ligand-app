# ==========================================================
# 04_build_complexes.py
# Memory-augmented complex construction with ligand provenance
#
# Every ligand in every generated complex carries its parent
# ligand + parent CCDC number. This information is propagated
# into generated_complexes.csv and later into elite_parents.csv.
# ==========================================================

import os
import random
import ast
import pandas as pd
from collections import Counter
import math

from ga_progress import write_progress

random.seed(42)

TARGET = 6
N_COMPLEXES = int(os.environ.get("N_COMPLEXES", 5000))
GEN = int(os.environ.get("GA_GEN", 0))

ALLOWED_PATTERNS = [
    (6,), (3,3), (4,1,1), (2,2,2),
    (1,2,3), (1,1,1,1,1,1), (5,1)
]

# ----------------------------------------------------------
# Load ligands + provenance
# ----------------------------------------------------------
df_real = pd.read_csv("ligand_donor_modes.csv")
df_mut = pd.read_csv("mutated_ligands.csv")

# donor mode map
real_modes = df_real.groupby("smiles")["donors"].apply(list).to_dict()
mut_modes = df_mut.groupby("smiles")["donors"].apply(list).to_dict()
MODE_MAP = {}
MODE_MAP.update(real_modes)
for lig, modes in mut_modes.items():
    MODE_MAP.setdefault(lig, [])
    MODE_MAP[lig] = sorted(set(MODE_MAP[lig]) | set(modes))

# provenance map for mutated ligands
PROV_MAP = {}
if not df_mut.empty:
    for lig, g in df_mut.groupby("smiles", sort=False):
        row = g.iloc[0]
        PROV_MAP[lig] = {
            "parent_ligand": str(row.get("parent_ligand", lig)),
            "parent_ccdc": str(row.get("parent_ccdc", "")),
            "parent_file_name": str(row.get("parent_file_name", "")),
            "generation": int(row.get("generation", GEN)),
            "mutation": str(row.get("mutation", "")),
        }

ligands = list(MODE_MAP.keys())

# ----------------------------------------------------------
# Add provenance for original database ligands
# ----------------------------------------------------------
def load_original_provenance():
    out = {}
    if not os.path.exists("GA.csv"):
        return out

    ga = pd.read_csv("GA.csv")
    if "CCDC" not in ga.columns:
        return out

    for _, row in ga.iterrows():
        ccdc = "" if pd.isna(row.get("CCDC")) else str(row.get("CCDC")).strip()
        source = "" if pd.isna(row.get("FileName")) else str(row.get("FileName")).strip()
        for i in range(1, 7):
            lig = row.get(f"L{i}")
            if not isinstance(lig, str) or lig.strip().upper() == "X":
                continue
            lig = lig.strip()
            out.setdefault(lig, {
                "parent_ligand": lig,
                "parent_ccdc": ccdc,
                "parent_file_name": source,
                "generation": 0,
                "mutation": "database_ligand",
            })
    return out

ORIGINAL_PROV = load_original_provenance()

for lig in ligands:
    if lig not in PROV_MAP:
        PROV_MAP[lig] = ORIGINAL_PROV.get(lig, {
            "parent_ligand": lig,
            "parent_ccdc": "",
            "parent_file_name": "",
            "generation": 0,
            "mutation": "database_ligand",
        })

# ----------------------------------------------------------
# Pattern weights (soft memory)
# ----------------------------------------------------------
pattern_weights = Counter({p: 1.0 for p in ALLOWED_PATTERNS})

if os.path.exists("elite_parents.csv"):
    elite = pd.read_csv("elite_parents.csv").sort_values("abs_err") if "abs_err" in pd.read_csv("elite_parents.csv", nrows=0).columns else pd.read_csv("elite_parents.csv")
    if not elite.empty:
        best = elite.iloc[0]
        try:
            best_pattern = tuple(sorted(ast.literal_eval(str(best["donor_list"]))))
            if best_pattern in pattern_weights:
                pattern_weights[best_pattern] = min(pattern_weights[best_pattern] + 1.5, 4.0)
        except Exception:
            pass

# ----------------------------------------------------------
# Sampling
# ----------------------------------------------------------
TEMP = 1.5
patterns = list(pattern_weights.keys())
weights = [math.exp(pattern_weights[p] / TEMP) for p in patterns]

rows = []
seen = set()
max_attempts = N_COMPLEXES * 100
attempts = 0
last_report = 0
write_progress(stage="complex_generation", stage_progress=0.0, complexes_generated=0,
                complexes_target=N_COMPLEXES,
                message=f"Building candidate complexes: 0/{N_COMPLEXES:,}...")

while len(rows) < N_COMPLEXES and attempts < max_attempts:
    attempts += 1
    if attempts - last_report >= 250:
        last_report = attempts
        frac = len(rows) / max(1, N_COMPLEXES)
        write_progress(stage_progress=min(0.99, frac), complexes_generated=len(rows),
                       complexes_target=N_COMPLEXES,
                       message=f"Building candidate complexes: {len(rows):,}/{N_COMPLEXES:,}...")
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

    lig_combo = [l for l, _ in chosen]
    donor_list = [d for _, d in chosen]
    key = (tuple(lig_combo), tuple(donor_list))
    if key in seen:
        continue
    seen.add(key)

    parent_ligands = []
    parent_ccdcs = []
    parent_file_names = []
    mutations = []

    for lig in lig_combo:
        p = PROV_MAP.get(lig, {})
        parent_ligands.append(str(p.get("parent_ligand", lig)))
        parent_ccdcs.append(str(p.get("parent_ccdc", "")))
        parent_file_names.append(str(p.get("parent_file_name", "")))
        mutations.append(str(p.get("mutation", "")))

    rows.append({
        "ligands": ";".join(lig_combo),
        "donor_list": str(donor_list),
        "donor_sum": TARGET,
        "parent_ligands": str(parent_ligands),
        "parent_ccdcs": str(parent_ccdcs),
        "parent_file_names": str(parent_file_names),
        "mutations": str(mutations),
        "generation": GEN,
    })

if len(rows) < N_COMPLEXES:
    print(f"[WARNING] Could generate only {len(rows)} unique complexes after {attempts} attempts.")

pd.DataFrame(rows).to_csv("generated_complexes.csv", index=False)
write_progress(stage="complex_generation", stage_progress=1.0, complexes_generated=len(rows),
                complexes_target=N_COMPLEXES,
                message=f"Candidate complex generation finished: {len(rows):,} complexes.")
print("[INFO] Generated complexes:", len(rows))

