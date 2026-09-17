import streamlit as st
import subprocess
import sys
import os
import ast
import html
import pandas as pd

from gdrive_save import (
    download_pipeline_from_drive,
    upload_pipeline_to_drive,
)

PYTHON = sys.executable
st.set_page_config(page_title="ZFS-driven Ligand SMILES Generator", layout="wide")

# ================= STYLES =================
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

st.title("🔬 ZFS-driven Ligand SMILES Generator")
st.write("GA + GNN oracle pipeline for target ZFS")

# ================= SIDEBAR =================
st.sidebar.header("🎯 Target settings")
target_zfs = st.sidebar.number_input("Target ZFS (cm⁻¹)", value=-180.0)
mode_label = st.sidebar.selectbox("Mode", ["X-ray", "DFT"])
mode = "crystal" if mode_label == "X-ray" else "optimized"
max_gen = st.sidebar.number_input("Max GA generations", 1, 1000, 5)
run = st.sidebar.button("🚀 Run")


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
    """Return compact per-ligand source labels.

    Only the information requested by the user is shown:
      - Mutated from reported ligand — CCDC XXXXXXX
      - Reported ligand — CCDC XXXXXXX
    Parent SMILES and mutation names are intentionally not displayed.
    """
    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))
    rows = []
    for i, _ in enumerate(ligands):
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else ""
        label = "Reported ligand" if mutation in {"", "database_ligand", "database_seed"} else "Mutated from reported ligand"
        rows.append((f"L{i+1}", label, ccdc or "not found"))
    return rows


def format_source_cell(row):
    parts = []
    for ligand_id, label, ccdc in source_rows(row):
        parts.append(
            f'<div class="source-item"><span class="source-label">{html.escape(ligand_id)}</span> · '
            f'{html.escape(label)} · <span class="ccdc">CCDC {html.escape(str(ccdc))}</span></div>'
        )
    return '<div class="source-cell">' + ''.join(parts) + '</div>' if parts else "—"


def render_result_table(row):
    ligands = html.escape(str(row.get("ligands", "")))
    donor_pattern = html.escape(str(row.get("donor_list", "")))
    donor_sum = html.escape(str(row.get("donor_sum", "")))
    zfs = float(row.get("zfs_pred", 0.0))
    ed = float(row.get("ed_pred", 0.0))
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
    """Compact final CCDC references, shown only after target achievement."""
    st.markdown("### 🧪 Follow these CCDC numbers for synthesis")
    records = []
    for ligand_id, label, ccdc in source_rows(row):
        records.append(
            f'<div class="final-row"><b>{html.escape(ligand_id)}</b> · '
            f'{html.escape(label)} · <b>CCDC {html.escape(str(ccdc))}</b></div>'
        )
    if records:
        st.markdown('<div class="final-box">' + ''.join(records) + '</div>', unsafe_allow_html=True)
    else:
        st.warning("No ligand/CCDC provenance was resolved for the final candidate.")


def load_checkpoint():
    path = "ga_checkpoint.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        r = df.iloc[0]
        saved_target = float(r.get("target_zfs", target_zfs))
        saved_mode = str(r.get("mode", mode)).strip().lower()
        if abs(saved_target - float(target_zfs)) > 1e-12 or saved_mode != mode:
            return None
        return {
            "next_generation": max(1, int(r.get("next_generation", 1))),
            "status": str(r.get("status", "running")).strip().lower(),
        }
    except Exception:
        return None


def save_checkpoint(next_generation, status="running"):
    pd.DataFrame([{
        "target_zfs": float(target_zfs),
        "mode": mode,
        "completed_generation": max(0, int(next_generation) - 1),
        "next_generation": int(next_generation),
        "status": status,
    }]).to_csv("ga_checkpoint.csv", index=False)


# ================= RUN =================
if run:
    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    st.info("Checking database...")
    db_ret = subprocess.call([PYTHON, "00_target_decision.py", str(target_zfs)])

    if db_ret == 0:
        st.success("🎯 Direct database match found")
        result = pd.read_csv("retrieved_solution.csv")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.stop()

    st.warning("⚠️ No suitable database hit found → 🚀 Entering AI-guided design mode")

    restored = download_pipeline_from_drive(target_zfs, mode_label)
    checkpoint = load_checkpoint() if restored else None

    if checkpoint and checkpoint["status"] == "target_achieved":
        st.success(f"♻️ Target was already achieved in Generation {checkpoint['next_generation'] - 1}. Restored final result.")
        if os.path.exists("elite_parents.csv"):
            elite = pd.read_csv("elite_parents.csv")
            if not elite.empty:
                best_row = elite.sort_values("abs_err").iloc[0]
                st.success(f"Best ZFS: {float(best_row['zfs_pred']):.2f} cm⁻¹")
                render_result_table(best_row)
                st.markdown("---")
                show_synthesis_references(best_row)
        st.stop()

    # `absolute_gen` is the persistent GA generation used internally.
    # `display_gen` is deliberately reset to 1 whenever the user presses Run.
    # This means a restored campaign continues from its saved population, while
    # the new visible run always starts at "Generation 1".
    if checkpoint:
        start_absolute_gen = checkpoint["next_generation"]
        first_run = False
        st.success("♻️ Previous campaign restored. Continuing from the saved population.")
    else:
        start_absolute_gen = 1
        first_run = True
        st.info("🆕 Initiating inverse molecular design")

    # max_gen means the number of generations requested for THIS Run click,
    # not an absolute generation number in the persistent campaign.
    end_absolute_gen = start_absolute_gen + int(max_gen) - 1
    progress = st.progress(0.0)
    final_best = None
    target_achieved = False

    # Re-use the same UI area instead of appending hundreds of tables.
    # This keeps the Streamlit page small/stable for long (e.g. 500-gen) runs.
    generation_view = st.empty()

    for absolute_gen in range(start_absolute_gen, end_absolute_gen + 1):
        display_gen = absolute_gen - start_absolute_gen + 1
        os.environ["GA_GEN"] = str(absolute_gen)
        progress.progress(min(1.0, display_gen / max(1, int(max_gen))))

        if absolute_gen == 1 and first_run:
            subprocess.call([PYTHON, "00_build_ligand_donor_map.py"])
            subprocess.call([PYTHON, "01_select_seeds.py"])
            subprocess.call([PYTHON, "02_extract_seed_ligands.py"])

        # GA stages. Provenance is retained in files but not printed here.
        ret = subprocess.call([PYTHON, "03_ligand_mutation.py"])
        if ret != 0:
            st.error(f"Ligand mutation failed in Generation {display_gen}.")
            break
        ret = subprocess.call([PYTHON, "04_build_complexes.py"])
        if ret != 0:
            st.error(f"Complex generation failed in Generation {display_gen}.")
            break
        ret = subprocess.call([PYTHON, "05_oracle_screen.py"])
        if ret != 0:
            st.error(f"Oracle screening failed in Generation {display_gen}.")
            break

        if os.path.exists("elite_parents.csv"):
            elite = pd.read_csv("elite_parents.csv")
            if not elite.empty:
                best_row = elite.sort_values("abs_err").iloc[0].copy()
                final_best = best_row
                D_value = float(best_row["zfs_pred"])
                ED_value = float(best_row["ed_pred"])

                with generation_view.container():
                    st.subheader(f"Generation {display_gen}")
                    st.success(f"Best ZFS so far: {D_value:.2f} cm⁻¹")
                    render_result_table(best_row)

                if D_value <= target_zfs:
                    target_achieved = True
                    save_checkpoint(absolute_gen + 1, "target_achieved")
                    upload_pipeline_to_drive(target_zfs, mode_label)
                    break

        # IMPORTANT: persist the next generation after every completed iteration.
        # If Streamlit Cloud restarts, the campaign resumes from this exact point
        # instead of starting again at Generation 1.
        save_checkpoint(absolute_gen + 1, "running")
        upload_pipeline_to_drive(target_zfs, mode_label)

    # ================= FINAL =================
    if target_achieved and final_best is not None:
        st.markdown("---")
        show_synthesis_references(final_best)

        st.markdown("### 📋 Final candidate")
        final_cols = {
            "Ligand Combination": final_best.get("ligands", ""),
            "Predicted D": final_best.get("zfs_pred", ""),
            "E/D": final_best.get("ed_pred", ""),
        }
        st.dataframe(pd.DataFrame([final_cols]), use_container_width=True, hide_index=True)
    elif final_best is not None:
        st.info(f"Completed {int(max_gen)} generations in this run. The latest population has been saved and can be continued with Run.")
