# ==========================================================
# 06_run_until_target.py
# MODE-aware driver (crystal / optimized)
# ==========================================================

import os
import sys
import pandas as pd

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
    print(f"\n===== GENERATION {gen} =====")
    os.environ["GA_GEN"] = str(gen)

    os.system("python 03_ligand_mutation.py")
    os.system("python 04_build_complexes.py")

    ret = os.system("python 05_oracle_screen.py")
    if ret != 0:
        print("❌ Oracle failed")
        sys.exit(1)

    elite = pd.read_csv("elite_parents.csv")
    best = elite["zfs_pred"].min()

    print("Best predicted ZFS:", best)

    if best <= TARGET:
        print("🎯 TARGET ACHIEVED")
        break
