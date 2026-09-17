# ==========================================================
# 03_ligand_mutation.py
# Reaction-based ligand mutation with full lineage
# Dataset: opt_D.csv
# Uses: opt_zfs
#
# CCDC ADDITION ONLY:
# CCDC provenance is carried from the reported parent complex
# into mutated_ligands.csv and mutation_lineage.csv.
# The original GA generation logic is unchanged.
# ==========================================================

import os
import random
import ast
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

def norm(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def ligand_combo(row):
    vals = []
    for i in range(1, 7):
        v = norm(row.get(f"L{i}"))
        if v and v.upper() != "X":
            vals.append(v)
    return tuple(vals)

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

# ----------------------------------------------------------
# CCDC lookup (separate file so original GA.csv is untouched)
# ----------------------------------------------------------
CCDC_BY_FILE = {}
if os.path.exists("CCDC_lookup.csv"):
    cdf = pd.read_csv("CCDC_lookup.csv")
    for _, r in cdf.iterrows():
        f = norm(r.get("FileName"))
        c = norm(r.get("CCDC"))
        if f and c:
            CCDC_BY_FILE[f] = c

# Map each reported ligand to a CCDC using the original opt_D anchor rows.
# FileName is preferred; exact ligand combination is the fallback.
CCDC_BY_LIGAND = {}
CCDC_BY_COMBO = {}
if os.path.exists("opt_D.csv"):
    _od = pd.read_csv("opt_D.csv")
    for _, r in _od.iterrows():
        f = norm(r.get("File Name"))
        ccdc = CCDC_BY_FILE.get(f, "")
        combo = ligand_combo(r)
        if combo and ccdc:
            CCDC_BY_COMBO[tuple(combo)] = ccdc
            for lig in combo:
                CCDC_BY_LIGAND.setdefault(lig, ccdc)

# Also make a ligand map directly from the CCDC dataset as a fallback.
if os.path.exists("GA.csv") and os.path.exists("CCDC_lookup.csv"):
    _ga = pd.read_csv("GA.csv")
    for _, r in _ga.iterrows():
        f = norm(r.get("FileName"))
        ccdc = CCDC_BY_FILE.get(f, "")
        if not ccdc:
            continue
        for lig in ligand_combo(r):
            CCDC_BY_LIGAND.setdefault(lig, ccdc)

# ----------------------------------------------------------
# Reaction SMARTS (aromatic C-H substitution)
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

ga_df["dist"] = (ga_df["opt_zfs"] - TARGET_ZFS).abs()
anchors = ga_df.sort_values("dist").head(K_ANCHORS)

parents = set()
PROV_MAP = {}
for _, row in anchors.iterrows():
    combo = ligand_combo(row)
    file_name = norm(row.get("File Name"))
    ccdc = CCDC_BY_FILE.get(file_name, "") or CCDC_BY_COMBO.get(tuple(combo), "")
    for lig in combo:
        if lig in MODE_MAP:
            parents.add(lig)
            PROV_MAP.setdefault(lig, {
                "parent_ligand": lig,
                "parent_ccdc": ccdc,
                "parent_file_name": file_name,
                "mutation": "database_ligand",
            })

# ----------------------------------------------------------
# Add elite parents (memory)
# ----------------------------------------------------------
if os.path.exists("elite_parents.csv"):
    elite = pd.read_csv("elite_parents.csv")
    for _, erow in elite.iterrows():
        ligs = parse_list(erow.get("ligands", ""))
        ccdcs = parse_list(erow.get("parent_ccdcs", ""))
        files = parse_list(erow.get("parent_file_names", ""))
        for j, lig in enumerate(ligs):
            if lig in MODE_MAP:
                parents.add(lig)
                ccdc = ccdcs[j] if j < len(ccdcs) else CCDC_BY_LIGAND.get(lig, "")
                fname = files[j] if j < len(files) else ""
                PROV_MAP[lig] = {
                    "parent_ligand": lig,
                    "parent_ccdc": ccdc,
                    "parent_file_name": fname,
                    "mutation": "elite_parent",
                }
        # Backward compatibility with an old elite_parents.csv that has no provenance.
        if not ligs:
            for lig in str(erow.get("ligands", "")).split(";"):
                lig = lig.strip()
                if lig in MODE_MAP:
                    parents.add(lig)
                    PROV_MAP.setdefault(lig, {
                        "parent_ligand": lig,
                        "parent_ccdc": CCDC_BY_LIGAND.get(lig, ""),
                        "parent_file_name": "",
                        "mutation": "database_ligand",
                    })

# Ensure every parent has a fallback CCDC record.
for lig in parents:
    PROV_MAP.setdefault(lig, {
        "parent_ligand": lig,
        "parent_ccdc": CCDC_BY_LIGAND.get(lig, ""),
        "parent_file_name": "",
        "mutation": "database_ligand",
    })

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

    source = PROV_MAP.get(p, {})
    parent_ccdc = source.get("parent_ccdc", "")
    parent_file_name = source.get("parent_file_name", "")

    for name, rxn in REACTIONS.items():
        m = aromatic_alkylation(p, rxn)
        if m:
            MODE_MAP[m] = MODE_MAP[p].copy()
            mutated.add(m)
            PROV_MAP[m] = {
                "parent_ligand": p,
                "parent_ccdc": parent_ccdc,
                "parent_file_name": parent_file_name,
                "mutation": name,
            }
            lineage.append({
                "parent": p,
                "child": m,
                "parent_ccdc": parent_ccdc,
                "parent_file_name": parent_file_name,
                "mutation": name,
                "generation": GEN
            })

    m = atom_type_mutation(p)
    if m:
        MODE_MAP[m] = MODE_MAP[p].copy()
        mutated.add(m)
        PROV_MAP[m] = {
            "parent_ligand": p,
            "parent_ccdc": parent_ccdc,
            "parent_file_name": parent_file_name,
            "mutation": "atom_type_substitution",
        }
        lineage.append({
            "parent": p,
            "child": m,
            "parent_ccdc": parent_ccdc,
            "parent_file_name": parent_file_name,
            "mutation": "atom_type_substitution",
            "generation": GEN
        })

    m = halogen_exchange(p)
    if m:
        MODE_MAP[m] = MODE_MAP[p].copy()
        mutated.add(m)
        PROV_MAP[m] = {
            "parent_ligand": p,
            "parent_ccdc": parent_ccdc,
            "parent_file_name": parent_file_name,
            "mutation": "halogen_exchange",
        }
        lineage.append({
            "parent": p,
            "child": m,
            "parent_ccdc": parent_ccdc,
            "parent_file_name": parent_file_name,
            "mutation": "halogen_exchange",
            "generation": GEN
        })

# ----------------------------------------------------------
# Save mutated ligands
# ----------------------------------------------------------
rows = []
for lig in mutated:
    meta = PROV_MAP.get(lig, {})
    for d in MODE_MAP.get(lig, []):
        rows.append({
            "smiles": lig,
            "donors": d,
            "parent_ligand": meta.get("parent_ligand", lig),
            "parent_ccdc": meta.get("parent_ccdc", CCDC_BY_LIGAND.get(lig, "")),
            "parent_file_name": meta.get("parent_file_name", ""),
            "mutation": meta.get("mutation", "database_ligand"),
            "generation": GEN,
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
