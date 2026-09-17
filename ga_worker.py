# ==========================================================
# ga_worker.py
# Long-running GA worker for the ZFS ligand generator.
# Streamlit is only the user interface; this process performs the GA.
# ==========================================================

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from ga_progress import write_progress
from gdrive_save import download_pipeline_from_drive, upload_pipeline_to_drive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "ga_checkpoint.csv")
STATUS_FILE = os.path.join(BASE_DIR, "ga_status.json")
LOCK_FILE = os.path.join(BASE_DIR, "ga_worker.lock")
STOP_FILE = os.path.join(BASE_DIR, "ga_stop.flag")

STOP_REQUESTED = False
START_TIME = time.time()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_status(**kwargs):
    # Keep worker-wide state while allowing stage scripts to add progress.
    current = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as fh:
                current = json.load(fh)
        except Exception:
            current = {}
    current.update(kwargs)
    current["updated_at"] = now_iso()
    current["elapsed_seconds"] = max(0.0, time.time() - START_TIME)
    atomic_json_write(STATUS_FILE, current)
    if kwargs.get("message"):
        write_progress(**kwargs, elapsed_seconds=current["elapsed_seconds"])


def save_checkpoint(target, mode, next_generation, status):
    pd.DataFrame([{
        "target_zfs": float(target),
        "mode": str(mode),
        "completed_generation": max(0, int(next_generation) - 1),
        "next_generation": int(next_generation),
        "status": str(status),
    }]).to_csv(CHECKPOINT, index=False)


def load_checkpoint(target, mode):
    if not os.path.exists(CHECKPOINT):
        return None
    try:
        df = pd.read_csv(CHECKPOINT)
        if df.empty:
            return None
        r = df.iloc[0]
        saved_target = float(r.get("target_zfs", target))
        saved_mode = str(r.get("mode", mode)).strip().lower()
        if abs(saved_target - float(target)) > 1e-12 or saved_mode != str(mode).lower():
            return None
        return {
            "next_generation": max(1, int(r.get("next_generation", 1))),
            "status": str(r.get("status", "running")).strip().lower(),
        }
    except Exception:
        return None


def clear_campaign_state():
    # Never delete the models, GA.csv, opt_D.csv, or application files.
    for name in [
        "ga_checkpoint.csv", "ga_status.json", "ga_stop.flag",
        "elite_parents.csv", "mutated_ligands.csv", "mutation_lineage.csv",
        "generated_complexes.csv", "oracle_screened_complexes.csv",
        "ligand_donor_modes.csv", "seed_complexes.csv", "seed_ligands.csv",
        "retrieved_solution.csv",
    ]:
        path = os.path.join(BASE_DIR, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def prepare_campaign_state(target, mode):
    """Use local state if it matches; otherwise restore matching Drive state."""
    local = load_checkpoint(target, mode)
    if local is not None:
        return "local"

    # No matching local checkpoint: old transient CSVs must not become the seed.
    clear_campaign_state()
    write_status(stage="restoring", state="running", stage_progress=0.05,
                 message="Checking Google Drive for a saved campaign...")
    try:
        restored = download_pipeline_from_drive(target, mode, include_optional=False)
        restored_checkpoint = load_checkpoint(target, mode)
        if restored and restored_checkpoint is not None:
            write_status(stage="restoring", state="running", stage_progress=1.0,
                         resume=True, message="Saved campaign restored from Google Drive.")
            return "drive"
        write_status(stage="restoring", state="running", stage_progress=1.0,
                     resume=False, message="No matching saved campaign was found. Starting a new campaign.")
    except Exception as exc:
        # Drive failure must not make the local GA unusable.
        write_status(stage="restoring", state="running", stage_progress=1.0,
                     resume=False, drive_message=f"Google Drive restore unavailable: {exc}",
                     message="Google Drive restore was unavailable. Starting a new campaign locally.")
    return "new"


def run_stage(script, env):
    cmd = [sys.executable, os.path.join(BASE_DIR, script)]
    print(f"[WORKER] Running {script}", flush=True)
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def request_stop(signum=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("[WORKER] Stop requested. Finishing the current generation...", flush=True)
    write_status(stage="stopping", message="Stop requested; finishing the current generation safely.")


def database_hit(target, mode):
    env = os.environ.copy()
    env["MODE"] = mode
    env["TARGET_ZFS"] = str(target)
    write_status(stage="database_check", state="running", stage_progress=0.1,
                 message="Checking the reported database for a direct target match...")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "00_target_decision.py"), str(target)],
        cwd=BASE_DIR, env=env,
    )
    write_status(stage="database_check", stage_progress=1.0,
                 message="Reported database check completed.")
    return result.returncode == 0


def read_best():
    path = os.path.join(BASE_DIR, "elite_parents.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "abs_err" not in df.columns:
            return None
        return df.sort_values("abs_err").iloc[0].to_dict()
    except Exception:
        return None


def sync_drive(target, mode, include_optional=False):
    try:
        upload_pipeline_to_drive(target, mode, include_optional=include_optional)
        return True, "Google Drive sync completed."
    except Exception as exc:
        print(f"[DRIVE] Sync failed: {exc}", flush=True)
        return False, f"Google Drive sync failed: {exc}"


def main():
    global STOP_REQUESTED, START_TIME
    START_TIME = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--mode", choices=["crystal", "optimized"], required=True)
    parser.add_argument("--max-generations", type=int, required=True)
    parser.add_argument("--drive-sync-every", type=int, default=int(os.environ.get("DRIVE_SYNC_EVERY", "5")))
    parser.add_argument("--n-complexes", type=int, default=int(os.environ.get("N_COMPLEXES", "5000")))
    args = parser.parse_args()

    target = float(args.target)
    mode = args.mode.lower()
    requested_generations = max(1, int(args.max_generations))
    drive_every = max(1, int(args.drive_sync_every))
    n_complexes = max(1, int(args.n_complexes))

    os.chdir(BASE_DIR)
    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target)
    os.environ["N_COMPLEXES"] = str(n_complexes)

    try:
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except Exception:
        pass

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # Replace STARTING:<launcher PID> with the worker PID immediately.
    with open(LOCK_FILE, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))

    try:
        write_status(
            pid=os.getpid(), target_zfs=target, mode=mode,
            requested_generations=requested_generations,
            display_generation=0, absolute_generation=0,
            complexes_target=n_complexes, complexes_generated=0,
            oracle_total=0, ed_pass=0, mutations_generated=0,
            progress=0.0, stage_progress=0.0,
            stage="starting", state="running", target_achieved=False,
            message="GA worker started. Preparing the campaign...",
        )

        # A direct database match is always checked before GA generation.
        if database_hit(target, mode):
            save_checkpoint(target, mode, 1, "target_achieved")
            write_status(stage="database_hit", state="completed", target_achieved=True,
                         progress=1.0, stage_progress=1.0,
                         message="A reported database structure already satisfies the target criterion.")
            return 0

        source = prepare_campaign_state(target, mode)
        checkpoint = load_checkpoint(target, mode)

        if checkpoint and checkpoint["status"] == "target_achieved":
            best = read_best()
            write_status(stage="target_achieved", state="completed", target_achieved=True,
                         progress=1.0,
                         display_generation=0,
                         best_zfs=float(best["zfs_pred"]) if best and "zfs_pred" in best else None,
                         best_ed=float(best["ed_pred"]) if best and "ed_pred" in best else None,
                         message="Target was already achieved in the saved campaign.")
            return 0

        start_gen = checkpoint["next_generation"] if checkpoint else 1
        first_campaign = checkpoint is None
        end_gen = start_gen + requested_generations - 1

        write_status(
            stage="initializing", state="running", display_generation=1,
            absolute_generation=start_gen, progress=0.0, stage_progress=0.0,
            resume=not first_campaign,
            message=("Resuming the saved campaign and preparing the next generation..."
                     if not first_campaign else "Initializing the new GA campaign..."),
        )

        env = os.environ.copy()
        env["MODE"] = mode
        env["TARGET_ZFS"] = str(target)
        env["N_COMPLEXES"] = str(n_complexes)

        if first_campaign:
            write_status(stage="initializing", stage_progress=0.15, message="Building ligand donor map...")
            run_stage("00_build_ligand_donor_map.py", env)
            write_status(stage="initializing", stage_progress=0.50, message="Selecting seed complexes...")
            run_stage("01_select_seeds.py", env)
            write_status(stage="initializing", stage_progress=0.80, message="Extracting seed ligands...")
            run_stage("02_extract_seed_ligands.py", env)
            write_status(stage="initializing", stage_progress=1.0, message="Initial GA population is ready.")
        elif not os.path.exists(os.path.join(BASE_DIR, "ligand_donor_modes.csv")):
            write_status(stage="initializing", stage_progress=0.25,
                         message="Rebuilding the ligand donor map for the resumed campaign...")
            run_stage("00_build_ligand_donor_map.py", env)
            write_status(stage="initializing", stage_progress=1.0,
                         message="Resume data is ready.")

        # Models are loaded once for the entire worker lifetime.
        write_status(stage="loading_models", stage_progress=0.0,
                     message="Loading GNN oracle models (one-time setup)...")
        try:
            from oracle_engine import OracleEngine
            oracle = OracleEngine(mode)
        except Exception as exc:
            raise RuntimeError(
                "The GNN oracle could not be loaded. Check the deployed Python "
                "environment and torch-geometric installation."
            ) from exc
        write_status(stage="loading_models", stage_progress=1.0,
                     message="GNN oracle models loaded and ready.")

        for absolute_gen in range(start_gen, end_gen + 1):
            if os.path.exists(STOP_FILE):
                STOP_REQUESTED = True
            if STOP_REQUESTED:
                save_checkpoint(target, mode, absolute_gen, "stopped")
                write_status(state="stopped", stage="stopped",
                             display_generation=max(1, absolute_gen - start_gen + 1),
                             absolute_generation=absolute_gen,
                             next_generation=absolute_gen,
                             message="Stopped safely. The completed generations are saved and the next Run can resume.")
                sync_drive(target, mode, include_optional=False)
                return 0

            display_gen = absolute_gen - start_gen + 1
            env["GA_GEN"] = str(absolute_gen)
            base_progress = (display_gen - 1) / requested_generations

            # ---- Mutation ----
            write_status(
                stage="mutation", state="running", display_generation=display_gen,
                absolute_generation=absolute_gen, progress=base_progress,
                stage_progress=0.0, mutations_generated=0,
                message=f"Generation {display_gen}: generating ligand mutations...",
            )
            run_stage("03_ligand_mutation.py", env)
            mutation_count = 0
            try:
                mutation_count = len(pd.read_csv(os.path.join(BASE_DIR, "mutated_ligands.csv")))
            except Exception:
                pass
            write_status(stage="mutation", stage_progress=1.0,
                         mutations_generated=mutation_count,
                         message=f"Generation {display_gen}: ligand mutation step completed ({mutation_count:,} ligand records).")

            # ---- Complex generation ----
            write_status(
                stage="complex_generation", state="running", display_generation=display_gen,
                absolute_generation=absolute_gen, progress=base_progress + 0.20 / requested_generations,
                stage_progress=0.0, complexes_generated=0, complexes_target=n_complexes,
                message=f"Generation {display_gen}: building candidate complexes...",
            )
            run_stage("04_build_complexes.py", env)
            generated_count = 0
            try:
                generated_count = len(pd.read_csv(os.path.join(BASE_DIR, "generated_complexes.csv")))
            except Exception:
                pass
            write_status(stage="complex_generation", stage_progress=1.0,
                         complexes_generated=generated_count,
                         message=f"Generation {display_gen}: {generated_count:,} candidate complexes generated.")

            # ---- Oracle ----
            write_status(
                stage="oracle", state="running", display_generation=display_gen,
                absolute_generation=absolute_gen, progress=base_progress + 0.40 / requested_generations,
                stage_progress=0.0, oracle_total=0, ed_pass=0,
                message=f"Generation {display_gen}: screening candidates with the GNN oracle...",
            )
            elite, screened = oracle.screen("generated_complexes.csv", target_zfs=target)

            ed_pass = len(screened) if screened is not None else 0
            try:
                total_generated = len(pd.read_csv(os.path.join(BASE_DIR, "generated_complexes.csv")))
            except Exception:
                total_generated = generated_count
            oracle_total = total_generated
            write_status(stage="oracle", stage_progress=1.0,
                         oracle_total=total_generated, ed_pass=ed_pass,
                         message=f"Generation {display_gen}: GNN screening completed; {len(elite):,} elite candidates retained.")

            best = None
            if not elite.empty:
                best = elite.sort_values("abs_err").iloc[0]
                best_zfs = float(best["zfs_pred"])
                best_ed = float(best["ed_pred"])
            else:
                best_zfs = None
                best_ed = None

            next_generation = absolute_gen + 1
            achieved = best is not None and best_zfs <= target
            save_checkpoint(target, mode, next_generation,
                            "target_achieved" if achieved else "running")

            write_status(
                stage="generation_complete",
                state="completed" if achieved else "running",
                display_generation=display_gen,
                absolute_generation=absolute_gen,
                next_generation=next_generation,
                progress=display_gen / requested_generations,
                stage_progress=1.0,
                best_zfs=best_zfs,
                best_ed=best_ed,
                complexes_generated=generated_count,
                complexes_target=n_complexes,
                oracle_total=total_generated,
                ed_pass=ed_pass,
                target_achieved=bool(achieved),
                message=(f"🎯 Target achieved in Generation {display_gen}." if achieved
                         else f"Generation {display_gen} completed. Preparing the next generation..."),
            )

            if achieved or display_gen % drive_every == 0:
                ok, msg = sync_drive(target, mode, include_optional=False)
                write_status(drive_last_sync=now_iso() if ok else None,
                             drive_message=msg)

            if achieved:
                return 0

        write_status(
            state="idle", stage="run_complete",
            display_generation=requested_generations,
            absolute_generation=end_gen,
            next_generation=end_gen + 1,
            progress=1.0, stage_progress=1.0,
            target_achieved=False,
            message=f"Completed {requested_generations} generations in this Run.",
        )
        ok, msg = sync_drive(target, mode, include_optional=False)
        write_status(drive_last_sync=now_iso() if ok else None, drive_message=msg)
        return 0

    except Exception as exc:
        traceback.print_exc()
        current = load_checkpoint(target, mode)
        next_gen = current["next_generation"] if current else 1
        try:
            save_checkpoint(target, mode, next_gen, "error")
        except Exception:
            pass
        write_status(state="error", stage="error", target_achieved=False,
                     message=f"Worker stopped with an error: {exc}",
                     error=traceback.format_exc())
        try:
            ok, msg = sync_drive(target, mode, include_optional=False)
            write_status(drive_last_sync=now_iso() if ok else None, drive_message=msg)
        except Exception:
            pass
        return 1
    finally:
        try:
            if os.path.exists(LOCK_FILE):
                with open(LOCK_FILE, "r", encoding="utf-8") as fh:
                    owner = fh.read().strip()
                if owner == str(os.getpid()):
                    os.remove(LOCK_FILE)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
