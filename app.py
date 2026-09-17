import streamlit as st
import subprocess
import sys
import os
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

# ================= RUN =================
if run:
    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    st.info("Checking database...")
    db_ret = subprocess.call([PYTHON, "00_target_decision.py", str(target_zfs)])

    if db_ret == 0:
        st.success("🎯 Direct database match found")
        result = pd.read_csv("retrieved_solution.csv")
        st.dataframe(result)
        if "CCDC" in result.columns:
            st.info("🧪 The CCDC column gives the literature/crystal-structure identifier for experimental procedure lookup.")
        st.stop()

    st.warning("⚠️ No suitable database hit found → 🚀 Entering AI-guided design mode")

    restored = download_pipeline_from_drive(target_zfs, mode_label)

    if restored:
        st.success("♻️ Resuming inverse molecular design")
        first_run = False
    else:
        st.info("🆕 Initiating inverse molecular design")
        first_run = True

    progress = st.progress(0)

    for gen in range(1, int(max_gen) + 1):
        os.environ["GA_GEN"] = str(gen)
        progress.progress(gen / max_gen)
        st.subheader(f"Generation {gen}")

        if gen == 1 and first_run:
            st.write("Building donor map")
            subprocess.call([PYTHON, "00_build_ligand_donor_map.py"])

            st.write("Selecting seed complexes with CCDC provenance")
            subprocess.call([PYTHON, "01_select_seeds.py"])

            st.write("Extracting seed ligands + parent CCDC")
            subprocess.call([PYTHON, "02_extract_seed_ligands.py"])

        st.write("Generating ligand mutations + tracking parent CCDC")
        subprocess.call([PYTHON, "03_ligand_mutation.py"])

        st.write("Building six-coordinate complexes + propagating CCDC provenance")
        subprocess.call([PYTHON, "04_build_complexes.py"])

        subprocess.call([PYTHON, "05_oracle_screen.py"])

        # ================= SHOW BEST RESULT =================
        if os.path.exists("elite_parents.csv"):
            elite = pd.read_csv("elite_parents.csv")

            if not elite.empty:
                best_row = elite.sort_values("abs_err").iloc[0]

                ligand_combo = best_row["ligands"]
                donor_list = best_row["donor_list"]
                donor_sum = best_row["donor_sum"]
                D_value = float(best_row["zfs_pred"])
                ED_value = float(best_row["ed_pred"])
                parent_ligands = best_row.get("parent_ligands", "")
                parent_ccdcs = best_row.get("parent_CCDC_for_experiment", best_row.get("parent_ccdcs", ""))
                mutations = best_row.get("mutations", "")

                st.success(f"Best ZFS so far: {D_value:.2f} cm⁻¹")

                result_df = pd.DataFrame([{
                    "Ligand Combination": ligand_combo,
                    "Parent Ligand(s)": parent_ligands,
                    "Parent CCDC(s)": parent_ccdcs,
                    "Mutation(s)": mutations,
                    "Donor Pattern": donor_list,
                    "Total Donors": donor_sum,
                    "Predicted D": D_value,
                    "E/D": ED_value,
                }])

                st.dataframe(result_df, use_container_width=True)

                st.markdown("### 🧪 Experimental procedure lookup")
                st.write(
                    "The **Parent CCDC(s)** identify the known complexes from which the starting "
                    "ligands were taken. Use these CCDC numbers to locate the corresponding "
                    "crystal-structure paper/supporting information and synthetic procedure."
                )

                if os.path.exists("mutation_lineage.csv"):
                    lineage = pd.read_csv("mutation_lineage.csv")
                    st.markdown("#### Complete ligand lineage")
                    st.dataframe(lineage.tail(100), use_container_width=True)

                if D_value <= target_zfs:
                    st.success("🎯 Target achieved")
                    upload_pipeline_to_drive(target_zfs, mode_label)
                    break

        # ================= SAVE STATE =================
        if any(os.path.exists(f) for f in [
            "mutated_ligands.csv",
            "mutation_lineage.csv",
            "generated_complexes.csv",
            "elite_parents.csv",
        ]):
            upload_pipeline_to_drive(target_zfs, mode_label)
            st.write("☁️ Design campaign checkpoint saved")
