# ==========================================================
# 03_ligand_mutation.py
# Reaction-based ligand mutation with full experimental lineage
#
# IMPORTANT:
# Every ligand carries the CCDC number of the database complex
# from which its parent ligand was obtained.
#
# Outputs:
#   mutated_ligands.csv
#   mutation_lineage.csv
#
# Lineage columns include:
#   parent_ligand, child_ligand, parent_ccdc,
#   parent_file_name, mutation, generation
# ==========================================================

import os
import random
import ast
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from ga_progress import write_progress

random.seed(42)

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
TARGET_ZFS = float(os.environ.get("TARGET_ZFS", -150))
GEN = int(os.environ.get("GA_GEN", 0))
K_ANCHORS = int(os.environ.get("K_ANCHORS", 15))
MODE = os.environ.get("MODE", "optimized").lower()

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


def norm_key(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def row_ligands(row):
    return [norm_key(row.get(f"L{i}")) for i in range(1, 7)]


def resolve_ccdc_from_ga(file_name, ligand_combo=None):
    """Resolve CCDC from the updated GA.csv.

    Priority:
      1. FileName match
      2. Exact six-ligand combination match
      3. No match -> blank
    """
    if not os.path.exists("GA.csv"):
        return ""

    ga = pd.read_csv("GA.csv")
    if "CCDC" not in ga.columns:
        return ""

    # 1. FileName is the strongest available cross-dataset key.
    if "FileName" in ga.columns and norm_key(file_name):
        hit = ga[ga["FileName"].astype(str).str.strip() == norm_key(file_name)]
        if len(hit):
            c = hit.iloc[0]["CCDC"]
            return "" if pd.isna(c) else str(c).strip()

    # 2. Exact ligand-combination match, ignoring X/NaN ordering.
    if ligand_combo is not None:
        target = sorted([x for x in ligand_combo if x and x.upper() != "X"])
        if target:
            for _, r in ga.iterrows():
                cand = sorted([x for x in row_ligands(r) if x and x.upper() != "X"])
                if cand == target:
                    c = r.get("CCDC", "")
                    return "" if pd.isna(c) else str(c).strip()

    return ""


def build_parent_metadata():
    """Return parent ligand -> all known CCDC/source records.

    The actual anchor row remains authoritative; this map is mainly used
    as a fallback for elite ligands generated in previous generations.
    """
    records = {}
    if not os.path.exists("GA.csv"):
        return records

    ga = pd.read_csv("GA.csv")
    for _, row in ga.iterrows():
        ccdc = row.get("CCDC", "")
        if pd.isna(ccdc):
            ccdc = ""
        else:
            ccdc = str(ccdc).strip()

        source = row.get("FileName", "")
        if pd.isna(source):
            source = ""
        else:
            source = str(source).strip()

        for i in range(1, 7):
            lig = row.get(f"L{i}")
            if not isinstance(lig, str) or lig.strip().upper() == "X":
                continue
            lig = lig.strip()
            records.setdefault(lig, []).append({
                "ccdc": ccdc,
                "file_name": source,
            })
    return records


# ----------------------------------------------------------
# Reaction SMARTS
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
# Parent/anchor selection
# ----------------------------------------------------------
parent_metadata = build_parent_metadata()

if MODE == "crystal":
    anchor_file = "GA.csv"
    zfs_col = "zfs"
    file_col = "FileName"
else:
    anchor_file = "opt_D.csv"
    zfs_col = "opt_zfs"
    file_col = "File Name"

db = pd.read_csv(anchor_file)
db[zfs_col] = pd.to_numeric(db[zfs_col], errors="coerce")
db = db.dropna(subset=[zfs_col]).copy()
db["dist"] = (db[zfs_col] - TARGET_ZFS).abs()
anchors = db.sort_values("dist").head(K_ANCHORS)

# parent_records is a list so that a ligand can retain its exact
# source CCDC even when the same SMILES occurs in multiple complexes.
parent_records = {}

for _, row in anchors.iterrows():
    file_name = row.get(file_col, "")
    ccdc = row.get("CCDC", "") if "CCDC" in db.columns else ""

    if pd.isna(ccdc) or not str(ccdc).strip():
        # opt_D does not contain CCDC in the supplied project.
        # Resolve it against the updated GA.csv.
        ccdc = resolve_ccdc_from_ga(file_name, row_ligands(row))
    else:
        ccdc = str(ccdc).strip()

    source_name = "" if pd.isna(file_name) else str(file_name).strip()

    for lig in row_ligands(row):
        if not lig or lig.upper() == "X":
            continue
        if lig not in MODE_MAP:
            continue
        parent_records.setdefault(lig, []).append({
            "ccdc": ccdc,
            "file_name": source_name,
        })

# ----------------------------------------------------------
# Add elite parents from previous generation
# ----------------------------------------------------------
if os.path.exists("elite_parents.csv"):
    elite = pd.read_csv("elite_parents.csv")
    for _, row in elite.iterrows():
        combo = str(row.get("ligands", ""))
        ccdc_list_raw = row.get("parent_ccdcs", "")
        parent_ligands_raw = row.get("parent_ligands", "")

        try:
            parent_ccdcs = ast.literal_eval(str(ccdc_list_raw))
        except Exception:
            parent_ccdcs = [x.strip() for x in str(ccdc_list_raw).split(";") if x.strip()]

        try:
            parent_ligands = ast.literal_eval(str(parent_ligands_raw))
        except Exception:
            parent_ligands = [x.strip() for x in str(parent_ligands_raw).split(";") if x.strip()]

        ligands = [x.strip() for x in combo.split(";") if x.strip()]
        for j, lig in enumerate(ligands):
            if lig not in MODE_MAP:
                continue
            ccdc = ""
            if j < len(parent_ccdcs):
                ccdc = str(parent_ccdcs[j]).strip()
            source_lig = lig
            if j < len(parent_ligands):
                source_lig = str(parent_ligands[j]).strip()

            parent_records.setdefault(lig, []).append({
                "ccdc": ccdc,
                "file_name": "elite_previous_generation",
                "parent_ligand": source_lig,
            })

# Deduplicate parent records.
for lig in list(parent_records):
    seen = set()
    clean = []
    for r in parent_records[lig]:
        key = (r.get("ccdc", ""), r.get("file_name", ""), r.get("parent_ligand", lig))
        if key not in seen:
            seen.add(key)
            clean.append(r)
    parent_records[lig] = clean

parents = sorted(parent_records)
print(f"[INFO] MODE = {MODE}")
print(f"[INFO] Parent ligands: {len(parents)}")
write_progress(stage="mutation", stage_progress=0.0, mutations_generated=0,
                message=f"Preparing ligand mutations from {len(parents):,} parent ligands...")


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
    new_symbol = random.choice(ATOM_MUTATIONS[a.GetSymbol()])
    a.SetAtomicNum(Chem.Atom(new_symbol).GetAtomicNum())
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

# Keep explicit metadata for every ligand.
ligand_metadata = {}
for p in parents:
    recs = parent_records.get(p, [{"ccdc": "", "file_name": ""}])
    chosen_rec = recs[0]
    ligand_metadata[p] = {
        "parent_ligand": chosen_rec.get("parent_ligand", p),
        "parent_ccdc": chosen_rec.get("ccdc", ""),
        "parent_file_name": chosen_rec.get("file_name", ""),
        "generation": GEN,
        "mutation": "database_seed" if chosen_rec.get("file_name", "") != "elite_previous_generation" else "elite_parent",
    }

for parent_index, p in enumerate(parents, start=1):
    source_recs = parent_records.get(p, [{"ccdc": "", "file_name": ""}])
    if parent_index == 1 or parent_index % 2 == 0 or parent_index == len(parents):
        frac = parent_index / max(1, len(parents))
        write_progress(stage_progress=min(0.99, frac),
                       mutations_generated=len(lineage),
                       message=f"Generating ligand mutations: parent {parent_index:,}/{len(parents):,}...")

    for source_rec in source_recs:
        parent_ccdc = source_rec.get("ccdc", "")
        parent_file_name = source_rec.get("file_name", "")
        true_parent = source_rec.get("parent_ligand", p)

        mutation_results = []
        for name, rxn in REACTIONS.items():
            mutation_results.append((name, aromatic_alkylation(p, rxn)))

        mutation_results.append(("atom_type_substitution", atom_type_mutation(p)))
        mutation_results.append(("halogen_exchange", halogen_exchange(p)))

        for mutation_name, child in mutation_results:
            if not child:
                continue

            MODE_MAP[child] = MODE_MAP.get(child, MODE_MAP[p].copy()).copy()
            mutated.add(child)

            child_meta = {
                "parent_ligand": true_parent,
                "parent_ccdc": parent_ccdc,
                "parent_file_name": parent_file_name,
                "generation": GEN,
                "mutation": mutation_name,
            }

            # Preserve first known provenance for the child; if the same
            # child is produced from multiple parents, all relationships
            # remain in mutation_lineage.csv.
            ligand_metadata.setdefault(child, child_meta)

            lineage.append({
                "parent_ligand": true_parent,
                "parent_ccdc": parent_ccdc,
                "parent_file_name": parent_file_name,
                "parent_smiles_used": p,
                "child_ligand": child,
                "mutation": mutation_name,
                "generation": GEN,
            })

# ----------------------------------------------------------
# Save mutated ligands with provenance
# ----------------------------------------------------------
rows = []
for lig in sorted(mutated):
    meta = ligand_metadata.get(lig, {
        "parent_ligand": lig,
        "parent_ccdc": "",
        "parent_file_name": "",
        "generation": GEN,
        "mutation": "unknown",
    })

    for d in sorted(MODE_MAP.get(lig, [])):
        rows.append({
            "smiles": lig,
            "donors": d,
            "parent_ligand": meta["parent_ligand"],
            "parent_ccdc": meta["parent_ccdc"],
            "parent_file_name": meta["parent_file_name"],
            "generation": meta["generation"],
            "mutation": meta["mutation"],
        })

pd.DataFrame(rows).to_csv("mutated_ligands.csv", index=False)

# ----------------------------------------------------------
# Save complete lineage history
# ----------------------------------------------------------
df_lineage = pd.DataFrame(lineage)
if os.path.exists("mutation_lineage.csv"):
    old = pd.read_csv("mutation_lineage.csv")
    df_lineage = pd.concat([old, df_lineage], ignore_index=True)

if not df_lineage.empty:
    df_lineage.drop_duplicates(inplace=True)

df_lineage.to_csv("mutation_lineage.csv", index=False)
write_progress(stage="mutation", stage_progress=1.0,
                mutations_generated=len(lineage),
                message=f"Ligand mutation step finished: {len(lineage):,} mutations generated.")

# ----------------------------------------------------------
# Console output for experimentalists
# ----------------------------------------------------------
print(f"\n[INFO] Generated {len(lineage)} ligand mutations in generation {GEN}.")
if lineage:
    print("[CCDC PROVENANCE]")
    for r in lineage:
        ccdc = r["parent_ccdc"] if r["parent_ccdc"] else "CCDC_NOT_FOUND"
        print(
            f"  {r['parent_ligand']}  --[{r['mutation']}]-->  {r['child_ligand']}  | Parent CCDC: {ccdc}"
        )
print(f"[INFO] Mutated ligands: {len(mutated)}")
