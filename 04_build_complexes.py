# ==========================================================
# 04_build_complexes.py
# Memory-augmented complex construction
# OUTPUT: generated_complexes.csv
# ==========================================================

import os
import random
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
# Pattern weights (soft memory)
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
# Sampling
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

    rows.append({
        "ligands": ";".join(l for l, _ in chosen),
        "donor_list": str([d for _, d in chosen]),
        "donor_sum": TARGET
    })

pd.DataFrame(rows).to_csv("generated_complexes.csv", index=False)
print("[INFO] Generated complexes:", len(rows))
