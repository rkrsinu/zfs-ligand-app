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
    parent_ccdcs = best_row.get("parent_CCDC_for_experiment", "")

    print(f"🏆 Best predicted ZFS: {best_zfs:.4f}")
    print(f"📐 Predicted E/D: {best_ed:.4f}")
    print(f"🧬 Parent CCDC(s): {parent_ccdcs}")
    print(f"🧬 Parent ligand(s): {best_row.get('parent_ligands', '')}")
    print(f"🧪 Mutation(s): {best_row.get('mutations', '')}")

    if best_zfs <= TARGET:
        print("\n🎯 TARGET ACHIEVED")
        print("\n================ EXPERIMENTAL LOOKUP ================")
        print(f"Parent CCDC(s): {parent_ccdcs}")
        print(f"Parent ligand(s): {best_row.get('parent_ligands', '')}")
        print(f"Generated ligand(s): {best_row.get('ligands', '')}")
        print(f"Mutation(s): {best_row.get('mutations', '')}")
        print("======================================================")
        break
else:
    print("\n⚠️ MAX_GEN reached without achieving target.")
