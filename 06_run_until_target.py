# ==========================================================
# 06_run_until_target.py
# MODE-aware resumable GA driver.
# The generation checkpoint prevents a restarted process from
# starting the campaign again at Generation 1.
# ==========================================================

import os
import sys
import subprocess
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
CHECKPOINT = os.path.join(BASE_DIR, "ga_checkpoint.csv")

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


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return None
    try:
        df = pd.read_csv(CHECKPOINT)
        if df.empty:
            return None
        r = df.iloc[0]
        if abs(float(r.get("target_zfs", TARGET)) - TARGET) > 1e-12:
            return None
        if str(r.get("mode", MODE)).strip().lower() != MODE:
            return None
        return {
            "next_generation": max(1, int(r.get("next_generation", 1))),
            "status": str(r.get("status", "running")).strip().lower(),
        }
    except Exception:
        return None


def save_checkpoint(next_generation, status="running"):
    pd.DataFrame([{
        "target_zfs": TARGET,
        "mode": MODE,
        "completed_generation": max(0, next_generation - 1),
        "next_generation": next_generation,
        "status": status,
    }]).to_csv(CHECKPOINT, index=False)


# ----------------------------------------------------------
# Database lookup
# ----------------------------------------------------------
ret = run_script("00_target_decision.py", TARGET)

if ret == 0:
    print("\n🎯 Solution retrieved directly from database")
    result = pd.read_csv(os.path.join(BASE_DIR, "retrieved_solution.csv"))
    print(result.to_string(index=False))
    sys.exit(0)

print("⚠️ No database match — switching to GA")

MAX_GEN = int(os.environ.get("MAX_GEN", 3000))
checkpoint = load_checkpoint()

if checkpoint and checkpoint["status"] == "target_achieved":
    start_gen = checkpoint["next_generation"]
    print(f"♻️ Target already achieved in Generation {start_gen - 1}; no restart required.")
    sys.exit(0)

start_gen = checkpoint["next_generation"] if checkpoint else 1
print(f"[INFO] Starting/resuming at Generation {start_gen}")

# ----------------------------------------------------------
# GA loop
# ----------------------------------------------------------
for gen in range(start_gen, MAX_GEN + 1):

    print("\n==============================")
    print(f"🚀 GENERATION {gen}")
    print("==============================")

    os.environ["GA_GEN"] = str(gen)

    if gen == 1 and not checkpoint:
        print("🔹 Building donor map")
        if run_script("00_build_ligand_donor_map.py") != 0:
            sys.exit(1)
        print("🔹 Selecting seed complexes")
        if run_script("01_select_seeds.py") != 0:
            sys.exit(1)
        print("🔹 Extracting seed ligands")
        if run_script("02_extract_seed_ligands.py") != 0:
            sys.exit(1)

    if run_script("03_ligand_mutation.py") != 0:
        sys.exit(1)
    if run_script("04_build_complexes.py") != 0:
        sys.exit(1)
    if run_script("05_oracle_screen.py") != 0:
        sys.exit(1)

    if not os.path.exists(os.path.join(BASE_DIR, "elite_parents.csv")):
        print("❌ elite_parents.csv NOT created → pipeline broken")
        sys.exit(1)

    elite = pd.read_csv(os.path.join(BASE_DIR, "elite_parents.csv"))
    if elite.empty:
        print("❌ elite_parents.csv is empty → no survivors")
        save_checkpoint(gen + 1, "running")
        continue

    best_row = elite.sort_values("abs_err").iloc[0]
    best_zfs = float(best_row["zfs_pred"])
    best_ed = float(best_row["ed_pred"])
    print(f"🏆 Best predicted ZFS: {best_zfs:.4f}")
    print(f"📐 Predicted E/D: {best_ed:.4f}")

    if best_zfs <= TARGET:
        print("\n🎯 TARGET ACHIEVED")
        save_checkpoint(gen + 1, "target_achieved")
        break

    # Persist immediately after a successful generation.
    save_checkpoint(gen + 1, "running")
    print(f"[CHECKPOINT] Next generation = {gen + 1}")
else:
    print("\n⚠️ MAX_GEN reached without achieving target.")
