# ==========================================================
# ga_worker.py
#
# Long-running GA worker for the ZFS ligand generator.
# Streamlit is NOT the computational engine anymore.
#
# Features
# --------
# * runs independently from the Streamlit request/session
# * loads GNN models once and reuses them for all generations
# * saves a local checkpoint after every completed generation
# * synchronizes only resume-critical files to Google Drive periodically
# * resumes from the saved internal generation
# * reports a human-readable generation counter starting at 1 for each Run
# * writes ga_status.json for the Streamlit UI
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

from gdrive_save import upload_pipeline_to_drive
from oracle_engine import OracleEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "ga_checkpoint.csv")
STATUS_FILE = os.path.join(BASE_DIR, "ga_status.json")
LOCK_FILE = os.path.join(BASE_DIR, "ga_worker.lock")

STOP_REQUESTED = False


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_status(**kwargs):
    current = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as fh:
                current = json.load(fh)
        except Exception:
            current = {}
    current.update(kwargs)
    current["updated_at"] = now_iso()
    atomic_json_write(STATUS_FILE, current)


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
        if abs(saved_target - float(target)) > 1e-12:
            return None
        if saved_mode != str(mode).lower():
            return None
        return {
            "next_generation": max(1, int(r.get("next_generation", 1))),
            "status": str(r.get("status", "running")).strip().lower(),
        }
    except Exception:
        return None


def run_stage(script, env):
    cmd = [sys.executable, os.path.join(BASE_DIR, script)]
    print(f"[WORKER] Running {script}", flush=True)
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def request_stop(signum=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("[WORKER] Stop requested. Finishing the current safe point...", flush=True)
    write_status(stage="stopping", message="Stop requested; worker will stop at the next generation boundary.")


def database_hit(target, mode):
    env = os.environ.copy()
    env["MODE"] = mode
    env["TARGET_ZFS"] = str(target)
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "00_target_decision.py"), str(target)],
        cwd=BASE_DIR,
        env=env,
    )
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
    global STOP_REQUESTED

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

    # A new worker run starts with a clean stop flag.
    try:
        if os.path.exists(os.path.join(BASE_DIR, "ga_stop.flag")):
            os.remove(os.path.join(BASE_DIR, "ga_stop.flag"))
    except Exception:
        pass

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # The launcher creates a temporary STARTING lock. Replace it immediately
    # with this worker PID so Streamlit can keep tracking the real worker even
    # if the original Streamlit session disappears.
    with open(LOCK_FILE, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))

    try:
        write_status(
            pid=os.getpid(),
            target_zfs=target,
            mode=mode,
            requested_generations=requested_generations,
            display_generation=0,
            absolute_generation=0,
            stage="starting",
            state="running",
            message="Starting GA worker...",
        )

        # Direct database check is performed once per new Run.
        write_status(stage="database_check", message="Checking the reported database...")
        if database_hit(target, mode):
            save_checkpoint(target, mode, 1, "target_achieved")
            write_status(
                stage="database_hit",
                state="completed",
                target_achieved=True,
                message="A reported database structure was found within the configured tolerance.",
            )
            return 0

        checkpoint = load_checkpoint(target, mode)

        if checkpoint and checkpoint["status"] == "target_achieved":
            best = read_best()
            write_status(
                stage="target_achieved",
                state="completed",
                target_achieved=True,
                absolute_generation=max(0, checkpoint["next_generation"] - 1),
                display_generation=0,
                best_zfs=float(best["zfs_pred"]) if best and "zfs_pred" in best else None,
                best_ed=float(best["ed_pred"]) if best and "ed_pred" in best else None,
                message="Target was already achieved in the saved campaign.",
            )
            return 0

        start_gen = checkpoint["next_generation"] if checkpoint else 1
        first_campaign = checkpoint is None
        end_gen = start_gen + requested_generations - 1

        write_status(
            stage="initializing",
            absolute_generation=start_gen,
            display_generation=1,
            resume=not first_campaign,
            message=(
                f"Resuming from internal Generation {start_gen}."
                if not first_campaign
                else "Initializing a new GA campaign."
            ),
        )

        env = os.environ.copy()
        env["MODE"] = mode
        env["TARGET_ZFS"] = str(target)
        env["N_COMPLEXES"] = str(n_complexes)

        if first_campaign:
            write_status(stage="initializing", message="Building ligand donor map...")
            run_stage("00_build_ligand_donor_map.py", env)
            if STOP_REQUESTED:
                save_checkpoint(target, mode, 1, "stopped")
                write_status(state="stopped", stage="stopped", message="Worker stopped before Generation 1.")
                return 0

            write_status(stage="initializing", message="Selecting seed complexes...")
            run_stage("01_select_seeds.py", env)
            write_status(stage="initializing", message="Extracting seed ligands...")
            run_stage("02_extract_seed_ligands.py", env)

        # A resumed campaign only needs the checkpoint + elite population.
        # Rebuild the donor map if the container was recreated and its local
        # transient files are gone. This avoids storing large CSVs on Drive.
        if not os.path.exists(os.path.join(BASE_DIR, "ligand_donor_modes.csv")):
            write_status(stage="initializing", message="Rebuilding ligand donor map for the resumed campaign...")
            run_stage("00_build_ligand_donor_map.py", env)

        # Load the four neural-network objects ONCE.
        write_status(stage="loading_models", message="Loading GNN oracle models once...")
        oracle = OracleEngine(mode)

        for absolute_gen in range(start_gen, end_gen + 1):
            if os.path.exists(os.path.join(BASE_DIR, "ga_stop.flag")):
                STOP_REQUESTED = True
            if STOP_REQUESTED:
                save_checkpoint(target, mode, absolute_gen, "stopped")
                write_status(
                    state="stopped",
                    stage="stopped",
                    absolute_generation=absolute_gen,
                    display_generation=max(1, absolute_gen - start_gen + 1),
                    message=f"Stopped. Next run will resume from internal Generation {absolute_gen}.",
                )
                return 0

            display_gen = absolute_gen - start_gen + 1
            env["GA_GEN"] = str(absolute_gen)

            write_status(
                stage="mutation",
                state="running",
                display_generation=display_gen,
                absolute_generation=absolute_gen,
                progress=(display_gen - 1) / requested_generations,
                message=f"Generating ligand mutations for Generation {display_gen}...",
            )
            run_stage("03_ligand_mutation.py", env)

            write_status(
                stage="complex_generation",
                display_generation=display_gen,
                absolute_generation=absolute_gen,
                progress=(display_gen - 0.5) / requested_generations,
                message=f"Generating candidate complexes for Generation {display_gen}...",
            )
            run_stage("04_build_complexes.py", env)

            write_status(
                stage="oracle",
                display_generation=display_gen,
                absolute_generation=absolute_gen,
                progress=(display_gen - 0.25) / requested_generations,
                message=f"Screening candidates with the GNN oracle for Generation {display_gen}...",
            )
            elite, _ = oracle.screen("generated_complexes.csv", target)

            best = None
            if not elite.empty:
                best = elite.sort_values("abs_err").iloc[0]
                best_zfs = float(best["zfs_pred"])
                best_ed = float(best["ed_pred"])
            else:
                best_zfs = None
                best_ed = None

            # Checkpoint immediately after the completed generation.
            next_generation = absolute_gen + 1
            achieved = best is not None and best_zfs <= target
            save_checkpoint(
                target,
                mode,
                next_generation,
                "target_achieved" if achieved else "running",
            )

            write_status(
                stage="generation_complete",
                state="completed" if achieved else "running",
                display_generation=display_gen,
                absolute_generation=absolute_gen,
                next_generation=next_generation,
                progress=display_gen / requested_generations,
                best_zfs=best_zfs,
                best_ed=best_ed,
                target_achieved=bool(achieved),
                message=(
                    f"Target achieved in Generation {display_gen}."
                    if achieved
                    else f"Generation {display_gen} completed."
                ),
            )

            # Drive is a BACKUP, not the per-generation storage engine.
            # Local checkpointing remains every generation.
            if achieved or display_gen % drive_every == 0:
                ok, msg = sync_drive(target, mode, include_optional=False)
                write_status(drive_last_sync=now_iso() if ok else None, drive_message=msg)

            if achieved:
                return 0

        # This Run finished its requested number of generations. The campaign
        # remains resumable because checkpoint points to the next internal gen.
        write_status(
            state="idle",
            stage="run_complete",
            display_generation=requested_generations,
            absolute_generation=end_gen,
            next_generation=end_gen + 1,
            target_achieved=False,
            message=(
                f"Completed {requested_generations} generations in this Run. "
                f"Next Run will resume from internal Generation {end_gen + 1}."
            ),
        )
        # Always create a final backup at the end of the Run.
        ok, msg = sync_drive(target, mode, include_optional=False)
        write_status(drive_last_sync=now_iso() if ok else None, drive_message=msg)
        return 0

    except Exception as exc:
        traceback.print_exc()
        write_status(
            state="error",
            stage="error",
            message=f"Worker failed: {exc}",
            error=traceback.format_exc(),
        )
        try:
            current = load_checkpoint(target, mode)
            next_gen = current["next_generation"] if current else 1
            save_checkpoint(target, mode, next_gen, "error")
            sync_drive(target, mode, include_optional=False)
        except Exception:
            pass
        return 1
    finally:
        try:
            if os.path.exists(LOCK_FILE):
                with open(LOCK_FILE, "r", encoding="utf-8") as fh:
                    owner = fh.read().strip()
                if owner in {"", str(os.getpid())}:
                    os.remove(LOCK_FILE)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
