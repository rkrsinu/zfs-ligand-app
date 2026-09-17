# ==========================================================
# 06_run_until_target.py
# MODE-aware GA driver with full ligand -> parent CCDC lineage.
# ==========================================================

import os
import sys
import subprocess
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

if len(sys.argv) < 3:
    print("Usage: python 06_run_until_target.py <TARGET_ZFS> <MODE>")
    print("MODE = crystal | optimized")
    sys.exit(2)

TARGET = float(sys.argv[1])
MODE = sys.argv[2].lower()

if MODE not in ["crystal", "optimized"]:
    print("❌ MODE must be crystal or optimized")
    sys.exit(2)

os.environ["MODE"] = MODE
os.environ["TARGET_ZFS"] = str(TARGET)

print(f"[INFO] MODE = {MODE}")
print(f"[INFO] TARGET ZFS = {TARGET}")


def run_script(name, *args):
    return subprocess.call([PYTHON, os.path.join(BASE_DIR, name), *map(str, args)], cwd=BASE_DIR)

# ----------------------------------------------------------
# Database lookup
# ----------------------------------------------------------
ret = run_script("00_target_decision.py", TARGET)

if ret == 0:
    print("\n🎯 Solution retrieved directly from database")
    result = pd.read_csv(os.path.join(BASE_DIR, "retrieved_solution.csv"))
    print(result.to_string(index=False))
    if "CCDC" in result.columns:
        print("\n[EXPERIMENT] Use the CCDC number above to locate the reported synthesis.")
    sys.exit(0)

print("⚠️ No database match — switching to GA")

MAX_GEN = int(os.environ.get("MAX_GEN", 3000))

# ----------------------------------------------------------
# GA loop
# ----------------------------------------------------------
for gen in range(1, MAX_GEN + 1):

    print("\n==============================")
    print(f"🚀 GENERATION {gen}")
    print("==============================")

    os.environ["GA_GEN"] = str(gen)

    if gen == 1:
        print("🔹 Building donor map")
        if run_script("00_build_ligand_donor_map.py") != 0:
            sys.exit(1)

        print("🔹 Selecting seed complexes")
        if run_script("01_select_seeds.py") != 0:
            sys.exit(1)

        print("🔹 Extracting seed ligands + CCDC lineage")
        if run_script("02_extract_seed_ligands.py") != 0:
            sys.exit(1)

    if run_script("03_ligand_mutation.py") != 0:
        sys.exit(1)

    if run_script("04_build_complexes.py") != 0:
        sys.exit(1)

    print("🔹 Oracle screening")
    if run_script("05_oracle_screen.py") != 0:
        print("❌ Oracle failed")
        sys.exit(1)

    if not os.path.exists(os.path.join(BASE_DIR, "elite_parents.csv")):
        print("❌ elite_parents.csv NOT created → pipeline broken")
        sys.exit(1)

    elite = pd.read_csv(os.path.join(BASE_DIR, "elite_parents.csv"))

    if elite.empty:
        print("❌ elite_parents.csv is empty → no survivors")
        sys.exit(1)

    best_row = elite.sort_values("abs_err").iloc[0]
    best_zfs = float(best_row["zfs_pred"])
    best_ed = float(best_row["ed_pred"])
    print(f"🏆 Best predicted ZFS: {best_zfs:.4f}")
    print(f"📐 Predicted E/D: {best_ed:.4f}")

    if best_zfs <= TARGET:
        parent_ccdcs = best_row.get("parent_CCDC_for_experiment", best_row.get("parent_ccdcs", ""))
        print("\n🎯 TARGET ACHIEVED")
        print("\n================ FOLLOW THESE CCDC NUMBERS FOR SYNTHESIS ================")

        try:
            import ast
            ligands = [x.strip() for x in str(best_row.get("ligands", "")).split(";") if x.strip()]
            parents = ast.literal_eval(str(best_row.get("parent_ligands", "[]")))
            ccdcs = ast.literal_eval(str(best_row.get("parent_ccdcs", "[]")))
            mutations = ast.literal_eval(str(best_row.get("mutations", "[]")))
        except Exception:
            ligands = [x.strip() for x in str(best_row.get("ligands", "")).split(";") if x.strip()]
            parents = [x.strip() for x in str(best_row.get("parent_ligands", "")).split(";") if x.strip()]
            ccdcs = [x.strip() for x in str(best_row.get("parent_ccdcs", "")).split(";") if x.strip()]
            mutations = [x.strip() for x in str(best_row.get("mutations", "")).split(";") if x.strip()]

        for i, ligand in enumerate(ligands):
            parent = parents[i] if i < len(parents) else ""
            ccdc = ccdcs[i] if i < len(ccdcs) else ""
            mutation = mutations[i] if i < len(mutations) else ""
            if mutation in ("", "database_ligand", "database_seed"):
                print(f"L{i+1}: Reported ligand — CCDC {ccdc or 'not found'}")
            else:
                print(f"L{i+1}: Mutated ligand — generated from the reported ligand {parent or 'not available'} — CCDC {ccdc or 'not found'}")

        print("========================================================================")
        break
else:
    print("\n⚠️ MAX_GEN reached without achieving target.")
