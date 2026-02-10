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

# USE opt_zfs (NOT zfs)
ga_df["dist"] = (ga_df["opt_zfs"] - TARGET_ZFS).abs()
anchors = ga_df.sort_values("dist").head(K_ANCHORS)

parents = set()
for _, row in anchors.iterrows():
    for i in range(1, 7):
        lig = row.get(f"L{i}")
        if isinstance(lig, str) and lig in MODE_MAP:
            parents.add(lig)

# ----------------------------------------------------------
# Add elite parents (memory)
# ----------------------------------------------------------
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
            lineage.append({
                "parent": p,
                "child": m,
                "mutation": name,
                "generation": GEN
            })

    m = atom_type_mutation(p)
    if m:
        MODE_MAP[m] = MODE_MAP[p].copy()
        mutated.add(m)
        lineage.append({
            "parent": p,
            "child": m,
            "mutation": "atom_type_substitution",
            "generation": GEN
        })

    m = halogen_exchange(p)
    if m:
        MODE_MAP[m] = MODE_MAP[p].copy()
        mutated.add(m)
        lineage.append({
            "parent": p,
            "child": m,
            "mutation": "halogen_exchange",
            "generation": GEN
        })

# ----------------------------------------------------------
# Save mutated ligands
# ----------------------------------------------------------
rows = []
for lig in mutated:
    for d in MODE_MAP.get(lig, []):
        rows.append({"smiles": lig, "donors": d})

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
