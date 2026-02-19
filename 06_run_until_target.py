# ==========================================================
# 06_run_until_target.py
# MODE-aware driver (crystal / optimized)
# ==========================================================

import os
import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Usage: python 06_run_until_target.py <TARGET_ZFS> <MODE>")
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

# ----------------------------------------------------------
# Database lookup
# ----------------------------------------------------------
ret = os.system(f"python 00_target_decision.py {TARGET}")

if ret == 0:
    print("\n🎯 Solution retrieved directly from database")
    print(pd.read_csv("retrieved_solution.csv"))
    sys.exit(0)

print("⚠️ No database match — switching to GA")

# ----------------------------------------------------------
# GA loop
# ----------------------------------------------------------
MAX_GEN = 3000

for gen in range(1, MAX_GEN + 1):

    print(f"\n==============================")
    print(f"🚀 GENERATION {gen}")
    print(f"==============================")

    os.environ["GA_GEN"] = str(gen)

    # ------------------------------------------------------
    # FIRST GENERATION → full pipeline
    # ------------------------------------------------------
    if gen == 1:

        print("🔹 Building donor map")
        os.system("python 00_build_ligand_donor_map.py")

        print("🔹 Selecting seed complexes")
        os.system("python 01_select_seeds.py")

        print("🔹 Extracting seed ligands")
        os.system("python 02_extract_seed_ligands.py")

    # ------------------------------------------------------
    # ALL GENERATIONS
    # ------------------------------------------------------
    print("🔹 Ligand mutation")
    os.system("python 03_ligand_mutation.py")

    print("🔹 Building complexes")
    os.system("python 04_build_complexes.py")

    print("🔹 Oracle screening")
    ret = os.system("python 05_oracle_screen.py")

    if ret != 0:
        print("❌ Oracle failed")
        sys.exit(1)

    # ------------------------------------------------------
    # ELITE CHECK
    # ------------------------------------------------------
    if not os.path.exists("elite_parents.csv"):
        print("❌ elite_parents.csv NOT created → pipeline broken")
        sys.exit(1)

    elite = pd.read_csv("elite_parents.csv")

    if elite.empty:
        print("❌ elite_parents.csv is empty → no survivors")
        sys.exit(1)

    best = elite["zfs_pred"].min()

    print(f"🏆 Best predicted ZFS so far: {best:.2f}")

    # ------------------------------------------------------
    # TARGET CHECK
    # ------------------------------------------------------
    if best <= TARGET:
        print("\n🎯 TARGET ACHIEVED")
        break
