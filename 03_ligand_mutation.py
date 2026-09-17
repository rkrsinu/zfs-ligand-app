# ==========================================================
# 03_ligand_mutation.py
# Reaction-based ligand mutation with full lineage
# Dataset: opt_D.csv
# Uses: opt_zfs
# ==========================================================

import os
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

random.seed(42)

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
TARGET_ZFS = float(os.environ.get("TARGET_ZFS", -150))
GEN = int(os.environ.get("GA_GEN", 0))
K_ANCHORS = 15

DONOR_ATOMS = {"N", "O", "S", "P", "Se"}
HALOGENS = ["F", "Cl", "Br", "I"]

ATOM_MUTATIONS = {
    "O": ["S", "Se"],
    "N": ["P"],
}

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def is_near_donor(atom):
    return any(n.GetSymbol() in DONOR_ATOMS for n in atom.GetNeighbors())

def safe_smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None

# ----------------------------------------------------------
# Reaction SMARTS (aromatic C–H substitution)
# ----------------------------------------------------------
REACTIONS = {
    "methyl_addition": rdChemReactions.ReactionFromSmarts("[cH:1]>>[c:1]C"),
    "ethyl_addition": rdChemReactions.ReactionFromSmarts("[cH:1]>>[c:1]CC"),
    "isopropyl_addition": rdChemReactions.ReactionFromSmarts("[cH:1]>>[c:1]C(C)C"),
}

# ----------------------------------------------------------
# Load ligand donor modes
# ----------------------------------------------------------
mode_df = pd.read_csv("ligand_donor_modes.csv")
MODE_MAP = mode_df.groupby("smiles")["donors"].apply(set).to_dict()

# ----------------------------------------------------------
# Parent ligand pool (ANCHORS from opt_D.csv)
# ----------------------------------------------------------
ga_df = pd.read_csv("opt_D.csv")

# CCDC provenance is read from the CCDC-enabled GA.csv.
# The mutation/selection logic below remains unchanged.
ccdc_df = pd.read_csv("GA.csv") if os.path.exists("GA.csv") else pd.DataFrame()

def _norm(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

# FileName -> CCDC is the primary cross-dataset mapping.
CCDC_BY_FILE = {}
if not ccdc_df.empty and "CCDC" in ccdc_df.columns and "FileName" in ccdc_df.columns:
    for _, _r in ccdc_df.iterrows():
        _fn = _norm(_r.get("FileName"))
        _cc = _norm(_r.get("CCDC"))
        if _fn and _cc:
            CCDC_BY_FILE[_fn] = _cc

def ccdc_for_anchor(row):
    """Return the reported CCDC for an opt_D anchor without changing GA logic."""
    fn = _norm(row.get("File Name"))
    if fn in CCDC_BY_FILE:
        return CCDC_BY_FILE[fn]

    # Fallback: exact ligand-combination match against GA.csv.
    if ccdc_df.empty or "CCDC" not in ccdc_df.columns:
        return ""
    target = sorted([_norm(row.get(f"L{i}")) for i in range(1, 7)
                     if _norm(row.get(f"L{i}")) and _norm(row.get(f"L{i}")).upper() != "X"])
    if not target:
        return ""
    for _, _r in ccdc_df.iterrows():
        cand = sorted([_norm(_r.get(f"L{i}")) for i in range(1, 7)
                       if _norm(_r.get(f"L{i}")) and _norm(_r.get(f"L{i}")).upper() != "X"])
        if cand == target:
            return _norm(_r.get("CCDC"))
    return ""

# USE opt_zfs (NOT zfs)
ga_df["dist"] = (ga_df["opt_zfs"] - TARGET_ZFS).abs()
anchors = ga_df.sort_values("dist").head(K_ANCHORS)

parents = set()
PARENT_CCDC = {}
for _, row in anchors.iterrows():
    anchor_ccdc = ccdc_for_anchor(row)
    for i in range(1, 7):
        lig = row.get(f"L{i}")
        if isinstance(lig, str) and lig in MODE_MAP:
            parents.add(lig)
            # Keep the first known reported CCDC for the ligand.
            if lig not in PARENT_CCDC:
                PARENT_CCDC[lig] = anchor_ccdc

# Add a fallback ligand -> CCDC mapping from GA.csv.
if not ccdc_df.empty and "CCDC" in ccdc_df.columns:
    for _, row in ccdc_df.iterrows():
        cc = _norm(row.get("CCDC"))
        if not cc:
            continue
        for i in range(1, 7):
            lig = row.get(f"L{i}")
            if isinstance(lig, str) and lig.strip() and lig.strip().upper() != "X" and lig.strip() in MODE_MAP:
                PARENT_CCDC.setdefault(lig.strip(), cc)

# ----------------------------------------------------------
# Add elite parents (memory)
# ----------------------------------------------------------
# Preserve the CCDC already attached to a mutated ligand when it
# becomes an elite parent in a later generation.
if os.path.exists("mutated_ligands.csv"):
    _old_mut = pd.read_csv("mutated_ligands.csv")
    if "parent_ccdc" in _old_mut.columns:
        for _, _r in _old_mut.iterrows():
            _lig = _norm(_r.get("smiles"))
            _cc = _norm(_r.get("parent_ccdc"))
            if _lig and _cc:
                PARENT_CCDC.setdefault(_lig, _cc)

if os.path.exists("elite_parents.csv"):
    elite = pd.read_csv("elite_parents.csv")
    for combo in elite["ligands"]:
        for lig in combo.split(";"):
            if lig in MODE_MAP:
                parents.add(lig)

parents = sorted(parents)
print("[INFO] Parent ligands:", len(parents))

# ----------------------------------------------------------
# Mutation operators
# ----------------------------------------------------------
def aromatic_alkylation(parent, rxn):
    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return None

    products = list(rxn.RunReactants((mol,)))
    random.shuffle(products)

    for prod_set in products:
        smi = safe_smiles(prod_set[0])
        if smi:
            return smi
    return None

def atom_type_mutation(parent):
    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return None

    rw = Chem.RWMol(mol)
    atoms = [
        a for a in rw.GetAtoms()
        if a.GetSymbol() in ATOM_MUTATIONS and not is_near_donor(a)
    ]

    if not atoms:
        return None

    a = random.choice(atoms)
    a.SetAtomicNum(
        Chem.Atom(random.choice(ATOM_MUTATIONS[a.GetSymbol()])).GetAtomicNum()
    )
    return safe_smiles(rw)

def halogen_exchange(parent):
    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return None

    rw = Chem.RWMol(mol)
    atoms = [
        a for a in rw.GetAtoms()
        if a.GetSymbol() in HALOGENS and not is_near_donor(a)
    ]

    if not atoms:
        return None

    a = random.choice(atoms)
    choices = [h for h in HALOGENS if h != a.GetSymbol()]
    a.SetAtomicNum(Chem.Atom(random.choice(choices)).GetAtomicNum())
    return safe_smiles(rw)

# ----------------------------------------------------------
# Run mutations + lineage
# ----------------------------------------------------------
mutated = set(parents)
lineage = []

for p in parents:

    for name, rxn in REACTIONS.items():
        m = aromatic_alkylation(p, rxn)
        if m:
            MODE_MAP[m] = MODE_MAP[p].copy()
            mutated.add(m)
            PARENT_CCDC.setdefault(m, PARENT_CCDC.get(p, ""))
            lineage.append({
                "parent": p,
                "child": m,
                "mutation": name,
                "generation": GEN,
                "parent_ccdc": PARENT_CCDC.get(p, "")
            })

    m = atom_type_mutation(p)
    if m:
        MODE_MAP[m] = MODE_MAP[p].copy()
        mutated.add(m)
        PARENT_CCDC.setdefault(m, PARENT_CCDC.get(p, ""))
        lineage.append({
            "parent": p,
            "child": m,
            "mutation": "atom_type_substitution",
            "generation": GEN,
            "parent_ccdc": PARENT_CCDC.get(p, "")
        })

    m = halogen_exchange(p)
    if m:
        MODE_MAP[m] = MODE_MAP[p].copy()
        mutated.add(m)
        PARENT_CCDC.setdefault(m, PARENT_CCDC.get(p, ""))
        lineage.append({
            "parent": p,
            "child": m,
            "mutation": "halogen_exchange",
            "generation": GEN,
            "parent_ccdc": PARENT_CCDC.get(p, "")
        })

# ----------------------------------------------------------
# Save mutated ligands
# ----------------------------------------------------------
rows = []
for lig in mutated:
    for d in MODE_MAP.get(lig, []):
        rows.append({
            "smiles": lig,
            "donors": d,
            "parent_ccdc": PARENT_CCDC.get(lig, "")
        })

pd.DataFrame(rows).to_csv("mutated_ligands.csv", index=False)

# ----------------------------------------------------------
# Save lineage
# ----------------------------------------------------------
df_lineage = pd.DataFrame(lineage)
if os.path.exists("mutation_lineage.csv"):
    df_lineage = pd.concat([pd.read_csv("mutation_lineage.csv"), df_lineage])

df_lineage.drop_duplicates(inplace=True)
df_lineage.to_csv("mutation_lineage.csv", index=False)

print("[INFO] Mutated ligands:", len(mutated))
print("[INFO] Lineage entries:", len(df_lineage))
