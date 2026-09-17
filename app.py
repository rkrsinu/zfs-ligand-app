# ==========================================================
# app.py
# ZFS-driven Ligand SMILES Generator
#
# Streamlit is the user interface/status monitor. The GA itself
# runs in ga_worker.py as a separate process.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
WORKER = os.path.join(BASE_DIR, "ga_worker.py")
CHECKPOINT = os.path.join(BASE_DIR, "ga_checkpoint.csv")
STATUS_FILE = os.path.join(BASE_DIR, "ga_status.json")
LOCK_FILE = os.path.join(BASE_DIR, "ga_worker.lock")
STOP_FILE = os.path.join(BASE_DIR, "ga_stop.flag")
LOG_FILE = os.path.join(BASE_DIR, "ga_worker.log")

# These are intentionally not exposed as main-page controls.
DEFAULT_N_COMPLEXES = int(os.environ.get("N_COMPLEXES", "5000"))
DEFAULT_DRIVE_SYNC_EVERY = int(os.environ.get("DRIVE_SYNC_EVERY", "5"))

st.set_page_config(page_title="ZFS-driven Ligand SMILES Generator", layout="wide")

st.markdown(
    """
<style>
.ga-table-wrap { width:100%; overflow-x:auto; border-radius:10px; }
.ga-table { width:100%; min-width:980px; border-collapse:collapse; table-layout:fixed;
            font-size:15px; background:transparent; }
.ga-table th { padding:13px 12px; text-align:left; font-weight:700;
               border-bottom:1px solid rgba(128,128,128,.35); }
.ga-table td { padding:14px 12px; vertical-align:middle;
               border-bottom:1px solid rgba(128,128,128,.20); }
.ga-table th:nth-child(1), .ga-table td:nth-child(1) { width:50%; }
.ga-table th:nth-child(2), .ga-table td:nth-child(2) { width:18%; }
.ga-table th:nth-child(3), .ga-table td:nth-child(3) { width:11%; text-align:center; }
.ga-table th:nth-child(4), .ga-table td:nth-child(4) { width:9%; text-align:center; }
.ga-table th:nth-child(5), .ga-table td:nth-child(5) { width:8%; text-align:right; }
.ga-table th:nth-child(6), .ga-table td:nth-child(6) { width:7%; text-align:right; }
.smiles-cell { white-space:normal; overflow-wrap:anywhere; word-break:break-word; line-height:1.45; }
.source-cell { line-height:1.45; white-space:normal; }
.source-item { margin:0 0 5px 0; }
.source-label { font-weight:650; }
.ccdc { font-weight:650; white-space:nowrap; }
.metric { font-variant-numeric:tabular-nums; }
.final-box { padding:16px 18px; border-radius:10px;
             border:1px solid rgba(128,128,128,.28); margin-top:8px; }
.final-row { padding:7px 0; border-bottom:1px solid rgba(128,128,128,.18); line-height:1.45; }
.final-row:last-child { border-bottom:0; }
.hero-status { padding:18px 20px; border-radius:14px;
               border:1px solid rgba(128,128,128,.28); margin:8px 0 16px 0; }
.stage-title { font-size:1.15rem; font-weight:700; margin-bottom:3px; }
.stage-sub { opacity:.72; font-size:.92rem; }
.activity { padding:10px 14px; border-radius:9px; border:1px solid rgba(128,128,128,.20); margin-top:8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔬 ZFS-driven Ligand SMILES Generator")
st.write("GA + GNN oracle pipeline for target ZFS")


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
    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))
    parents = parse_list(row.get("parent_ligands", ""))
    rows = []
    for i, lig in enumerate(ligands):
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else ""
        parent = parents[i] if i < len(parents) else ""
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
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )


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
    if not os.path.exists(LOCK_FILE):
        return False, None, None, None
    try:
        raw = open(LOCK_FILE, "r", encoding="utf-8").read().strip()
    except Exception:
        raw = ""

    status = read_json(STATUS_FILE)
    target = status.get("target_zfs")
    mode = status.get("mode")

    if raw.startswith("STARTING:"):
        launcher_pid = raw.split(":", 1)[1]
        if pid_alive(launcher_pid):
            return True, target, mode, None
    else:
        try:
            pid = int(raw)
        except Exception:
            pid = None
        if pid is not None and pid_alive(pid):
            return True, target, mode, pid

    try:
        os.remove(LOCK_FILE)
    except Exception:
        pass
    return False, None, None, None


def status_matches_target(status, target, mode):
    try:
        return (
            abs(float(status.get("target_zfs")) - float(target)) <= 1e-12
            and str(status.get("mode", "")).lower() == str(mode).lower()
        )
    except Exception:
        return False


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


def write_initial_status(target, mode, requested_generations):
    payload = {
        "pid": None,
        "target_zfs": float(target),
        "mode": mode,
        "requested_generations": int(requested_generations),
        "display_generation": 0,
        "absolute_generation": 0,
        "progress": 0.0,
        "stage_progress": 0.0,
        "complexes_target": DEFAULT_N_COMPLEXES,
        "complexes_generated": 0,
        "oracle_total": 0,
        "ed_pass": 0,
        "mutations_generated": 0,
        "state": "running",
        "stage": "starting",
        "target_achieved": False,
        "message": "Starting the GA worker...",
        "elapsed_seconds": 0.0,
        "activity": [f"{time.strftime('%H:%M:%S')} · Starting the GA worker..."] ,
        "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, STATUS_FILE)


def start_worker(target, mode, max_generations):
    running, active_target, active_mode, _ = worker_info()
    if running:
        if active_target is not None and active_mode is not None:
            return False, f"A GA campaign is already running for target {active_target:g} ({active_mode})."
        return False, "A GA worker is already starting. Please wait a few seconds."

    try:
        with open(LOCK_FILE, "x", encoding="utf-8") as fh:
            fh.write(f"STARTING:{os.getpid()}")
    except FileExistsError:
        return False, "A GA worker is already starting. Please wait a few seconds."

    # Write the target immediately so the monitor never displays an old
    # campaign while the worker process is being launched.
    write_initial_status(target, mode, max_generations)

    env = os.environ.copy()
    env["MODE"] = mode
    env["TARGET_ZFS"] = str(float(target))
    env["N_COMPLEXES"] = str(DEFAULT_N_COMPLEXES)
    env["DRIVE_SYNC_EVERY"] = str(DEFAULT_DRIVE_SYNC_EVERY)

    try:
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except Exception:
        pass

    log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    log_fh.write("\n\n================ NEW GA RUN ================\n")
    log_fh.write(
        f"Started: {time.ctime()} | target={target} | mode={mode} | "
        f"requested_gen={max_generations} | complexes/gen={DEFAULT_N_COMPLEXES}\n"
    )
    log_fh.flush()

    try:
        kwargs = dict(
            cwd=BASE_DIR,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        subprocess.Popen(
            [
                PYTHON, WORKER,
                "--target", str(float(target)),
                "--mode", mode,
                "--max-generations", str(int(max_generations)),
                "--drive-sync-every", str(DEFAULT_DRIVE_SYNC_EVERY),
                "--n-complexes", str(DEFAULT_N_COMPLEXES),
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


# ================= SIDEBAR =================
st.sidebar.header("🎯 Target settings")
target_zfs = st.sidebar.number_input("Target ZFS (cm⁻¹)", value=-180.0, step=1.0)
mode_label = st.sidebar.selectbox("Mode", ["X-ray", "DFT"])
mode = "crystal" if mode_label == "X-ray" else "optimized"
max_gen = st.sidebar.number_input("Max GA generations", min_value=1, max_value=1000, value=500, step=1)

run = st.sidebar.button("🚀 Run", type="primary", use_container_width=True)
stop = st.sidebar.button("⏹ Stop after current generation", use_container_width=True)

if stop:
    running, _, _, _ = worker_info()
    if running:
        request_stop()
        st.sidebar.success("Stop requested. The worker will stop after the current generation is safely completed.")
    else:
        st.sidebar.info("No GA worker is currently running.")

if run:
    running, active_target, active_mode, _ = worker_info()
    if running:
        if (
            active_target is not None
            and active_mode is not None
            and abs(float(active_target) - float(target_zfs)) <= 1e-12
            and str(active_mode).lower() == mode
        ):
            st.info("♻️ This campaign is already running. Live progress is shown below.")
        else:
            st.error(
                f"Another GA campaign is currently running (target={active_target}, mode={active_mode}). "
                "Stop it before starting a different campaign."
            )
    else:
        try:
            started, message = start_worker(float(target_zfs), mode, int(max_gen))
            if started:
                st.success("🚀 GA campaign started. Live progress will appear below.")
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
        if running and active_target is not None:
            st.warning(
                f"A different campaign is running (target={active_target}, mode={active_mode}). "
                "Select the same target and mode to view its progress."
            )
        else:
            st.info("Set the target and click **Run** to start or resume the GA campaign.")
        return

    state = str(status.get("state", "idle"))
    stage = str(status.get("stage", "idle"))
    display_gen = int(status.get("display_generation", 0) or 0)
    requested = int(status.get("requested_generations", max_gen) or max_gen)
    best_zfs = status.get("best_zfs")
    best_ed = status.get("best_ed")
    message = str(status.get("message", ""))
    progress = float(status.get("progress", 0.0) or 0.0)
    achieved = bool(status.get("target_achieved", False))
    complexes_generated = int(status.get("complexes_generated", 0) or 0)
    complexes_total = int(status.get("complexes_target", DEFAULT_N_COMPLEXES) or DEFAULT_N_COMPLEXES)
    ed_pass = int(status.get("ed_pass", 0) or 0)
    oracle_total = int(status.get("oracle_total", 0) or 0)
    mutations = int(status.get("mutations_generated", 0) or 0)
    elapsed = float(status.get("elapsed_seconds", 0.0) or 0.0)
    drive_msg = status.get("drive_message")
    last_update = status.get("updated_at", "—")

    if best_zfs is not None:
        try:
            target_error = abs(float(best_zfs) - float(target_zfs))
        except Exception:
            target_error = None
    else:
        target_error = None

    # Main progress area: no implementation details such as internal generation.
    st.markdown('<div class="hero-status">', unsafe_allow_html=True)
    if achieved:
        title = "🎯 Target achieved"
    elif state == "error":
        title = "❌ Calculation stopped because of an error"
    elif state == "stopped":
        title = "⏹ Calculation stopped safely"
    elif state == "idle" and stage == "run_complete":
        title = "✅ Run completed"
    elif running:
        title = "⚙️ GA calculation in progress"
    else:
        title = "GA campaign status"

    st.markdown(f'<div class="stage-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-sub">{html.escape(message or stage.replace("_", " ").title())}</div>', unsafe_allow_html=True)

    if requested > 0:
        st.progress(max(0.0, min(1.0, progress)), text=f"Overall progress · Generation {display_gen}/{requested}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generation", f"{display_gen}/{requested}" if display_gen else "Preparing…")
    c2.metric("Best predicted D", f"{float(best_zfs):.2f}" if best_zfs is not None else "—")
    c3.metric("Target distance", f"{target_error:.2f} cm⁻¹" if target_error is not None else "—")
    c4.metric("E/D", f"{float(best_ed):.4f}" if best_ed is not None else "—")

    st.markdown("#### 📊 Calculation progress")
    p1, p2, p3 = st.columns(3)
    p1.metric("Complexes generated", f"{complexes_generated:,}/{complexes_total:,}")
    p2.metric("Candidates screened", f"{oracle_total:,}")
    p3.metric("Passed E/D ≤ 0.22", f"{ed_pass:,}")

    p4, p5, p6 = st.columns(3)
    p4.metric("Ligand mutations", f"{mutations:,}")
    p5.metric("Elapsed time", format_elapsed(elapsed))
    p6.metric("Last update", str(last_update).replace("T", " ")[:19])

    # Stage-specific progress gives the user something visible even when a
    # single generation takes several minutes.
    stage_progress = status.get("stage_progress")
    if stage_progress is not None:
        try:
            stage_progress = max(0.0, min(1.0, float(stage_progress)))
            st.progress(stage_progress, text=f"Current step · {stage.replace('_', ' ').title()} · {stage_progress*100:.0f}%")
        except Exception:
            pass

    if drive_msg:
        if str(drive_msg).lower().startswith("google drive sync completed"):
            st.caption(f"☁️ {drive_msg}")
        else:
            st.caption(f"☁️ {drive_msg}")

    if stage in {"starting", "database_check", "restoring", "initializing", "loading_models"}:
        st.info("⏳ Preparing the calculation. The worker is active; the next progress values will appear automatically.")

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

    if state == "error":
        with st.expander("Worker error details", expanded=True):
            st.code(str(status.get("error", "No traceback available.")))

    best = load_latest_best()
    if best is not None:
        label = "Latest best candidate"
        if display_gen:
            label += f" · Generation {display_gen}"
        st.markdown(f"### 🧬 {label}")
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

    # Short activity feed from worker-written messages.
    history = status.get("activity", [])
    if history:
        st.markdown("#### 📝 Latest activity")
        for item in history[-6:]:
            st.markdown(
                f'<div class="activity">{html.escape(str(item))}</div>',
                unsafe_allow_html=True,
            )



def format_elapsed(seconds):
    try:
        seconds = max(0, int(float(seconds)))
    except Exception:
        return "—"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


live_monitor()
