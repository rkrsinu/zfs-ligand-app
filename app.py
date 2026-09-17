import streamlit as st
import subprocess
import sys
import os
import ast
import pandas as pd

from gdrive_save import (
    download_pipeline_from_drive,
    upload_pipeline_to_drive,
)

PYTHON = sys.executable
st.set_page_config(page_title="ZFS-driven Ligand SMILES Generator", layout="wide")

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
    """Safely convert stored Python-list strings or semicolon lists to a list."""
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


def format_ligand_origins(row):
    """Return compact per-ligand provenance for display in GA result tables."""
    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    parents = parse_list(row.get("parent_ligands", ""))
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))

    items = []
    for i, _ in enumerate(ligands):
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else ""
        is_database = mutation in {"", "database_ligand", "database_seed"}
        if is_database:
            label = "Reported ligand"
        else:
            label = "Mutated from reported ligand"
        items.append(f"L{i+1}: {label} — CCDC {ccdc or 'not found'}")
    return "<br>".join(items)


def show_synthesis_references(row):
    """Show only the parent-ligand/CCDC information needed for synthesis.

    This section is intentionally shown only after the target ZFS is achieved.
    For a mutated ligand, identify the reported parent ligand and its CCDC.
    For a database ligand, identify it simply as a reported ligand and give its CCDC.
    """
    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    parents = parse_list(row.get("parent_ligands", ""))
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))

    st.markdown("### 🧪 Ligand mutation & parent CCDC")
    st.write("Reported ligand source and parent CCDC.")

    records = []
    for i, child in enumerate(ligands):
        parent = parents[i] if i < len(parents) else ""
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else ""

        is_database = mutation in {"", "database_ligand", "database_seed"}
        if is_database:
            text = f"L{i+1}: Reported ligand — CCDC {ccdc or 'not found'}"
        else:
            text = (
                f"L{i+1}: Mutated ligand — generated from the reported ligand "
                f"{parent or 'not available'} — CCDC {ccdc or 'not found'}"
            )
        records.append({"Ligand": f"L{i+1}", "Synthesis reference": text})

    if records:
        for r in records:
            st.write(f"- **{r['Ligand']}:** {r['Synthesis reference'].split(': ', 1)[1]}")
    else:
        st.warning("No ligand/CCDC provenance was resolved for the final candidate.")


def show_experimental_lookup(row):
    """Backward-compatible wrapper for the final synthesis-reference section."""
    show_synthesis_references(row)


# ================= RUN =================
if run:
    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    st.info("Checking database...")
    db_ret = subprocess.call([PYTHON, "00_target_decision.py", str(target_zfs)])

    if db_ret == 0:
        st.success("🎯 Direct database match found")
        result = pd.read_csv("retrieved_solution.csv")
        st.dataframe(result, use_container_width=True)
        if "CCDC" in result.columns:
            st.markdown("### 🧪 Ligand mutation & parent CCDC")
            st.write("Reported ligand source and parent CCDC.")
        st.stop()

    st.warning("⚠️ No suitable database hit found → 🚀 Entering AI-guided design mode")

    restored = download_pipeline_from_drive(target_zfs, mode_label)
    first_run = not restored
    if restored:
        st.success("♻️ Resuming inverse molecular design")
    else:
        st.info("🆕 Initiating inverse molecular design")

    progress = st.progress(0)
    final_best = None

    for gen in range(1, int(max_gen) + 1):
        os.environ["GA_GEN"] = str(gen)
        progress.progress(gen / max_gen)
        st.subheader(f"Generation {gen}")

        if gen == 1 and first_run:
            subprocess.call([PYTHON, "00_build_ligand_donor_map.py"])
            subprocess.call([PYTHON, "01_select_seeds.py"])
            subprocess.call([PYTHON, "02_extract_seed_ligands.py"])

        # Run the pipeline silently here; detailed provenance is shown below.
        subprocess.call([PYTHON, "03_ligand_mutation.py"])
        subprocess.call([PYTHON, "04_build_complexes.py"])
        subprocess.call([PYTHON, "05_oracle_screen.py"])

        if os.path.exists("elite_parents.csv"):
            elite = pd.read_csv("elite_parents.csv")
            if not elite.empty:
                best_row = elite.sort_values("abs_err").iloc[0]
                final_best = best_row.copy()

                ligand_combo = best_row["ligands"]
                donor_list = best_row["donor_list"]
                donor_sum = best_row["donor_sum"]
                D_value = float(best_row["zfs_pred"])
                ED_value = float(best_row["ed_pred"])

                st.success(f"Best ZFS so far: {D_value:.2f} cm⁻¹")

                result_df = pd.DataFrame([{
                    "Ligand Combination": ligand_combo,
                    "Ligand source / CCDC": format_ligand_origins(best_row),
                    "Donor Pattern": donor_list,
                    "Total Donors": donor_sum,
                    "Predicted D": D_value,
                    "E/D": ED_value,
                }])
                st.markdown(
                    result_df.to_html(index=False, escape=False),
                    unsafe_allow_html=True,
                )

                # Do not show provenance, mutation details, structures, or synthesis
                # references during intermediate generations. They are shown only
                # after the target is achieved.
                if D_value <= target_zfs:
                    st.success("🎯 Target achieved")
                    upload_pipeline_to_drive(target_zfs, mode_label)
                    break

        if any(os.path.exists(f) for f in [
            "mutated_ligands.csv",
            "mutation_lineage.csv",
            "generated_complexes.csv",
            "elite_parents.csv",
        ]):
            upload_pipeline_to_drive(target_zfs, mode_label)
            st.write("☁️ Design campaign checkpoint saved")

    # ======================================================
    # FINAL SYNTHESIS REFERENCES — ONLY AFTER TARGET ACHIEVEMENT
    # ======================================================
    if final_best is not None:
        final_zfs = float(final_best.get("zfs_pred", 0.0))
        if final_zfs <= target_zfs:
            st.markdown("---")
            st.markdown("# 🧪 Ligand source & parent CCDC")
            show_synthesis_references(final_best)

            st.markdown("### 📋 Final candidate")
            final_cols = {
                "Ligand Combination": final_best.get("ligands", ""),
                "Predicted D": final_best.get("zfs_pred", ""),
                "E/D": final_best.get("ed_pred", ""),
            }
            st.dataframe(pd.DataFrame([final_cols]), use_container_width=True, hide_index=True)
