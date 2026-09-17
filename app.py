# ==========================================================
# app.py
# ZFS-driven Ligand SMILES Generator
#
# IMPORTANT ARCHITECTURE CHANGE
# -----------------------------
# Streamlit is now the USER INTERFACE / STATUS VIEWER only.
# The long-running GA is executed by ga_worker.py as a separate
# process. Therefore a Streamlit rerun/session refresh does not
# interrupt the current generation loop.
#
# The worker:
#   * loads GNN models once
#   * checkpoints locally every generation
#   * syncs only resume-critical files to Google Drive periodically
#   * resumes from the saved internal generation
#
# The visible generation counter resets to Generation 1 whenever
# the user starts a new Run click, even when the worker internally
# resumes from Generation 19, 50, etc.
# ==========================================================

import ast
import html
import json
import os
import subprocess
import sys
import time

import pandas as pd
import streamlit as st

from gdrive_save import download_pipeline_from_drive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
WORKER = os.path.join(BASE_DIR, "ga_worker.py")
CHECKPOINT = os.path.join(BASE_DIR, "ga_checkpoint.csv")
STATUS_FILE = os.path.join(BASE_DIR, "ga_status.json")
LOCK_FILE = os.path.join(BASE_DIR, "ga_worker.lock")
STOP_FILE = os.path.join(BASE_DIR, "ga_stop.flag")
LOG_FILE = os.path.join(BASE_DIR, "ga_worker.log")

st.set_page_config(page_title="ZFS-driven Ligand SMILES Generator", layout="wide")

# ================= STYLES =================
st.markdown(
    """
<style>
.ga-table-wrap { width: 100%; overflow-x: auto; border-radius: 10px; }
.ga-table { width: 100%; min-width: 980px; border-collapse: collapse; table-layout: fixed;
            font-size: 15px; background: transparent; }
.ga-table th { padding: 13px 12px; text-align: left; font-weight: 700;
               border-bottom: 1px solid rgba(128,128,128,.35); }
.ga-table td { padding: 14px 12px; vertical-align: middle;
               border-bottom: 1px solid rgba(128,128,128,.20); }
.ga-table th:nth-child(1), .ga-table td:nth-child(1) { width: 50%; }
.ga-table th:nth-child(2), .ga-table td:nth-child(2) { width: 18%; }
.ga-table th:nth-child(3), .ga-table td:nth-child(3) { width: 11%; text-align:center; }
.ga-table th:nth-child(4), .ga-table td:nth-child(4) { width: 9%; text-align:center; }
.ga-table th:nth-child(5), .ga-table td:nth-child(5) { width: 8%; text-align:right; }
.ga-table th:nth-child(6), .ga-table td:nth-child(6) { width: 7%; text-align:right; }
.smiles-cell { white-space: normal; overflow-wrap: anywhere; word-break: break-word;
               line-height: 1.45; }
.source-cell { line-height: 1.45; white-space: normal; }
.source-item { margin: 0 0 5px 0; }
.source-label { font-weight: 650; }
.ccdc { font-weight: 650; white-space: nowrap; }
.metric { font-variant-numeric: tabular-nums; }
.final-box { padding: 16px 18px; border-radius: 10px;
             border: 1px solid rgba(128,128,128,.28); margin-top: 8px; }
.final-row { padding: 7px 0; border-bottom: 1px solid rgba(128,128,128,.18); line-height: 1.45; }
.final-row:last-child { border-bottom: 0; }
.status-box { padding: 12px 15px; border-radius: 10px; border: 1px solid rgba(128,128,128,.28); }
.small-muted { opacity: .72; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔬 ZFS-driven Ligand SMILES Generator")
st.write("GA + GNN oracle pipeline for target ZFS")

# ================= HELPERS =================
def parse_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [x.strip() for x in text.split(";") if x.strip()]


def source_rows(row):
    """Compact provenance shown in result tables.

    Display is deliberately limited to:
      L1 · Reported ligand · CCDC XXXXXXX
      L2 · Mutated from reported ligand · CCDC XXXXXXX
    """
    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))
    parents = parse_list(row.get("parent_ligands", ""))

    rows = []
    for i, lig in enumerate(ligands):
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else ""
        parent = parents[i] if i < len(parents) else ""

        # A ligand is considered directly reported when it is the same
        # SMILES as its reported parent and no mutation operator was applied.
        is_reported = (
            mutation in {"", "database_ligand", "database_seed", "elite_parent"}
            and (not parent or parent == lig)
        )
        label = "Reported ligand" if is_reported else "Mutated from reported ligand"
        rows.append((f"L{i+1}", label, ccdc or "not found"))
    return rows


def format_source_cell(row):
    parts = []
    for ligand_id, label, ccdc in source_rows(row):
        parts.append(
            f'<div class="source-item"><span class="source-label">{html.escape(ligand_id)}</span> · '
            f'{html.escape(label)} · <span class="ccdc">CCDC {html.escape(str(ccdc))}</span></div>'
        )
    return '<div class="source-cell">' + "".join(parts) + "</div>" if parts else "—"


def render_result_table(row):
    try:
        zfs = float(row.get("zfs_pred", 0.0))
    except Exception:
        zfs = 0.0
    try:
        ed = float(row.get("ed_pred", 0.0))
    except Exception:
        ed = 0.0

    ligands = html.escape(str(row.get("ligands", "")))
    donor_pattern = html.escape(str(row.get("donor_list", "")))
    donor_sum = html.escape(str(row.get("donor_sum", "")))
    source = format_source_cell(row)

    table = f"""
    <div class="ga-table-wrap">
      <table class="ga-table">
        <thead><tr>
          <th>Ligand Combination</th>
          <th>Ligand source / CCDC</th>
          <th>Donor Pattern</th>
          <th>Total Donors</th>
          <th>Predicted D</th>
          <th>E/D</th>
        </tr></thead>
        <tbody><tr>
          <td class="smiles-cell">{ligands}</td>
          <td>{source}</td>
          <td>{donor_pattern}</td>
          <td class="metric">{donor_sum}</td>
          <td class="metric">{zfs:.4f}</td>
          <td class="metric">{ed:.5f}</td>
        </tr></tbody>
      </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)


def show_synthesis_references(row):
    st.markdown("### 🧪 Follow these CCDC numbers for synthesis")
    records = []
    for ligand_id, label, ccdc in source_rows(row):
        records.append(
            f'<div class="final-row"><b>{html.escape(ligand_id)}</b> · '
            f'{html.escape(label)} · <b>CCDC {html.escape(str(ccdc))}</b></div>'
        )
    if records:
        st.markdown('<div class="final-box">' + "".join(records) + "</div>", unsafe_allow_html=True)
    else:
        st.warning("No ligand/CCDC provenance was resolved for the final candidate.")


def read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def pid_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def worker_info():
    """Return (running, target, mode, pid)."""
    if not os.path.exists(LOCK_FILE):
        return False, None, None, None

    pid = None
    try:
        raw = open(LOCK_FILE, "r", encoding="utf-8").read().strip()
        if raw.startswith("STARTING:"):
            # A launcher is active; treat it as running briefly.
            launcher_pid = raw.split(":", 1)[1]
            return pid_alive(launcher_pid), None, None, None
        pid = int(raw)
    except Exception:
        pid = None

    if pid is not None and pid_alive(pid):
        status = read_json(STATUS_FILE)
        return True, status.get("target_zfs"), status.get("mode"), pid

    # Stale lock after a crash/reboot.
    try:
        os.remove(LOCK_FILE)
    except Exception:
        pass
    return False, None, None, None


def local_checkpoint_matches(target, mode):
    if not os.path.exists(CHECKPOINT):
        return False
    try:
        df = pd.read_csv(CHECKPOINT)
        if df.empty:
            return False
        r = df.iloc[0]
        return (
            abs(float(r.get("target_zfs", target)) - float(target)) <= 1e-12
            and str(r.get("mode", mode)).strip().lower() == str(mode).lower()
        )
    except Exception:
        return False


def clear_stale_local_files_for_new_campaign():
    """Prevent an unrelated old campaign from being mistaken for a new one."""
    for name in [
        "ga_checkpoint.csv",
        "ga_status.json",
        "elite_parents.csv",
        "mutated_ligands.csv",
        "mutation_lineage.csv",
        "generated_complexes.csv",
        "oracle_screened_complexes.csv",
        "ligand_donor_modes.csv",
        "seed_complexes.csv",
        "seed_ligands.csv",
        "retrieved_solution.csv",
    ]:
        path = os.path.join(BASE_DIR, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def prepare_campaign(target, mode):
    """Restore a matching campaign from Drive if no matching local state exists."""
    if local_checkpoint_matches(target, mode):
        return "local"

    # Do not overwrite an unrelated active campaign.
    running, active_target, active_mode, _ = worker_info()
    if running:
        return "active"

    clear_stale_local_files_for_new_campaign()

    try:
        restored = download_pipeline_from_drive(target, mode, include_optional=False)
        return "drive" if restored else "new"
    except Exception as exc:
        # A missing Drive configuration should not prevent local execution.
        st.warning(f"Google Drive restore was not available: {exc}")
        return "new"


def start_worker(target, mode, max_generations):
    running, active_target, active_mode, _ = worker_info()
    if running:
        return False, f"A GA worker is already running (target={active_target}, mode={active_mode})."

    # Atomic launch lock prevents double-clicks from creating two workers.
    try:
        with open(LOCK_FILE, "x", encoding="utf-8") as fh:
            fh.write(f"STARTING:{os.getpid()}")
    except FileExistsError:
        return False, "A GA worker is already starting. Please wait a few seconds."

    env = os.environ.copy()
    env["MODE"] = mode
    env["TARGET_ZFS"] = str(float(target))
    env["N_COMPLEXES"] = str(int(os.environ.get("N_COMPLEXES", "5000")))

    # Remove a stop request from an earlier run.
    try:
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except Exception:
        pass

    log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    log_fh.write("\n\n================ NEW GA RUN ================\n")
    log_fh.write(f"Started: {time.ctime()} | target={target} | mode={mode} | max_gen={max_generations}\n")
    log_fh.flush()

    try:
        kwargs = {
            "cwd": BASE_DIR,
            "env": env,
            "stdout": log_fh,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }
        # On Windows this detaches the child from the console; on Linux/Cloud
        # the normal child process is sufficient and survives Streamlit reruns.
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        subprocess.Popen(
            [
                PYTHON,
                WORKER,
                "--target",
                str(float(target)),
                "--mode",
                mode,
                "--max-generations",
                str(int(max_generations)),
                "--drive-sync-every",
                os.environ.get("DRIVE_SYNC_EVERY", "5"),
                "--n-complexes",
                os.environ.get("N_COMPLEXES", "5000"),
            ],
            **kwargs,
        )
        return True, "GA worker started."
    except Exception:
        try:
            log_fh.close()
        except Exception:
            pass
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass
        raise


def request_stop():
    with open(STOP_FILE, "w", encoding="utf-8") as fh:
        fh.write(str(time.time()))


def load_latest_best():
    path = os.path.join(BASE_DIR, "elite_parents.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        if "abs_err" in df.columns:
            df = df.sort_values("abs_err")
        return df.iloc[0]
    except Exception:
        return None


def status_matches_target(status, target, mode):
    try:
        return (
            abs(float(status.get("target_zfs")) - float(target)) <= 1e-12
            and str(status.get("mode", "")).lower() == str(mode).lower()
        )
    except Exception:
        return False


# ================= SIDEBAR =================
st.sidebar.header("🎯 Target settings")
target_zfs = st.sidebar.number_input("Target ZFS (cm⁻¹)", value=-180.0, step=1.0)
mode_label = st.sidebar.selectbox("Mode", ["X-ray", "DFT"])
mode = "crystal" if mode_label == "X-ray" else "optimized"
max_gen = st.sidebar.number_input("Max GA generations", min_value=1, max_value=1000, value=5, step=1)
n_complexes = st.sidebar.number_input("Complexes per generation", min_value=100, max_value=20000, value=5000, step=100)
drive_every = st.sidebar.number_input("Google Drive backup every N generations", min_value=1, max_value=50, value=5, step=1)

# These values are passed to the worker through the environment.
os.environ["N_COMPLEXES"] = str(int(n_complexes))
os.environ["DRIVE_SYNC_EVERY"] = str(int(drive_every))

run = st.sidebar.button("🚀 Run", type="primary", use_container_width=True)
stop = st.sidebar.button("⏹ Stop after current generation", use_container_width=True)

if stop:
    running, _, _, _ = worker_info()
    if running:
        request_stop()
        st.sidebar.success("Stop requested. The worker will stop at the next safe generation boundary.")
    else:
        st.sidebar.info("No GA worker is currently running.")

if run:
    running, active_target, active_mode, _ = worker_info()
    if running:
        if (
            active_target is not None
            and abs(float(active_target) - float(target_zfs)) <= 1e-12
            and str(active_mode).lower() == mode
        ):
            st.info("♻️ This target is already running. The page will continue to show its live status.")
        else:
            st.error(
                f"Another GA campaign is currently running (target={active_target}, mode={active_mode}). "
                "Stop it before starting a different campaign."
            )
    else:
        with st.spinner("Restoring saved campaign state..."):
            source = prepare_campaign(float(target_zfs), mode)
        try:
            started, message = start_worker(float(target_zfs), mode, int(max_gen))
            if started:
                if source == "drive":
                    st.success("♻️ Saved campaign restored from Google Drive. Starting the new Run from that state.")
                elif source == "local":
                    st.success("♻️ Saved local campaign restored. Starting the new Run from that state.")
                else:
                    st.success("🆕 New GA campaign started.")
            else:
                st.warning(message)
        except Exception as exc:
            st.error(f"Could not start GA worker: {exc}")


# ================= LIVE MONITOR =================
@st.fragment(run_every="3s")
def live_monitor():
    status = read_json(STATUS_FILE)
    running, active_target, active_mode, pid = worker_info()

    if not status_matches_target(status, float(target_zfs), mode):
        if running:
            st.warning(
                f"A different campaign is running (target={active_target}, mode={active_mode}). "
                "Select the same target/mode to monitor it."
            )
        else:
            st.info("Set the target and press **Run** to start or resume the GA campaign.")
        return

    state = str(status.get("state", "idle"))
    stage = str(status.get("stage", "idle"))
    display_gen = int(status.get("display_generation", 0) or 0)
    absolute_gen = int(status.get("absolute_generation", 0) or 0)
    next_gen = status.get("next_generation")
    requested = int(status.get("requested_generations", max_gen) or max_gen)
    best_zfs = status.get("best_zfs")
    best_ed = status.get("best_ed")
    message = str(status.get("message", ""))
    progress = float(status.get("progress", 0.0) or 0.0)
    achieved = bool(status.get("target_achieved", False))

    st.markdown('<div class="status-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run generation", f"{display_gen}/{requested}" if display_gen else "—")
    c2.metric("Internal generation", str(absolute_gen) if absolute_gen else "—")
    c3.metric("Best predicted D", f"{float(best_zfs):.2f}" if best_zfs is not None else "—")
    c4.metric("E/D", f"{float(best_ed):.4f}" if best_ed is not None else "—")
    st.markdown("</div>", unsafe_allow_html=True)

    if stage == "database_hit":
        st.success("🎯 A reported database structure already satisfies the target criterion.")
        path = os.path.join(BASE_DIR, "retrieved_solution.csv")
        if os.path.exists(path):
            try:
                result = pd.read_csv(path)
                st.dataframe(result, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Could not display the retrieved database solution: {exc}")
        return

    if state == "running":
        st.progress(max(0.0, min(1.0, progress)))
        st.info(f"⚙️ {message}")
    elif achieved:
        st.success(f"🎯 {message}")
    elif state == "idle" and stage == "run_complete":
        st.success(f"✅ {message}")
    elif state == "stopped":
        st.warning(f"⏹ {message}")
    elif state == "error":
        st.error(f"❌ {message}")
        with st.expander("Worker error details"):
            st.code(str(status.get("error", "No traceback available.")))
    else:
        st.info(message or f"Worker stage: {stage}")

    # Show only the latest generation result. This prevents the browser DOM
    # from growing to hundreds of tables during a long run.
    best = load_latest_best()
    if best is not None:
        st.markdown(f"### Generation {display_gen if display_gen else absolute_gen} — latest best candidate")
        render_result_table(best)

    if achieved and best is not None:
        st.markdown("---")
        show_synthesis_references(best)
        st.markdown("### 📋 Final candidate")
        final_cols = {
            "Ligand Combination": best.get("ligands", ""),
            "Predicted D": best.get("zfs_pred", ""),
            "E/D": best.get("ed_pred", ""),
        }
        st.dataframe(pd.DataFrame([final_cols]), use_container_width=True, hide_index=True)

    drive_message = status.get("drive_message")
    if drive_message:
        st.caption(drive_message)

    if running:
        st.caption(f"Worker PID: {pid} · Last update: {status.get('updated_at', '—')}")
    elif next_gen:
        st.caption(f"Next internal generation on the next Run: {next_gen}")


live_monitor()

# ================= NOTES =================
st.markdown("---")
st.caption(
    "The visible Run generation always starts at 1. If a saved campaign resumes internally from a later generation, "
    "the worker continues from that checkpoint while the visible counter resets for the new Run click."
)
