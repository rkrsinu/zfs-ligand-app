import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import time

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

mode = st.sidebar.selectbox(
    "Mode",
    ["crystal", "optimized"]
)

max_gen = st.sidebar.number_input(
    "Max GA generations",
    1,
    1000,
    5
)

run = st.sidebar.button("🚀 Run")

# ================= RUN =================

if run:

    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    st.info("Checking database...")

    db_ret = subprocess.call([PYTHON, "00_target_decision.py", str(target_zfs)])

    if db_ret == 0:

        st.success("🎯 Direct database match found")

        df = pd.read_csv("retrieved_solution.csv")

        if "FileName" in df.columns:
            df["FileName"] = df["FileName"].astype(str).str.replace("a", "", regex=False)

        st.dataframe(df)

        st.stop()

    st.warning("No valid database match → starting GA")

    # ================= RESTORE =================

    restored = download_pipeline_from_drive(target_zfs, mode)

    if restored:
        st.success("♻️ Resuming previous GA run")
        first_run = False
    else:
        st.info("🆕 Fresh GA run")
        first_run = True

    progress = st.progress(0)

    # ================= GA LOOP =================

    for gen in range(1, int(max_gen) + 1):

        os.environ["GA_GEN"] = str(gen)

        progress.progress(gen / max_gen)

        st.subheader(f"Generation {gen}")

        if gen == 1 and first_run:

            st.write("Building donor map")
            subprocess.call([PYTHON, "00_build_ligand_donor_map.py"])

            st.write("Selecting seed complexes")
            subprocess.call([PYTHON, "01_select_seeds.py"])

            st.write("Extracting seed ligands")
            subprocess.call([PYTHON, "02_extract_seed_ligands.py"])

        subprocess.call([PYTHON, "03_ligand_mutation.py"])
        subprocess.call([PYTHON, "04_build_complexes.py"])
        subprocess.call([PYTHON, "05_oracle_screen.py"])

        # ================= WAIT UNTIL ORACLE FINISHES =================

        zfs_col = None
        df = None

        for _ in range(10):  # try for 10 seconds

            if os.path.exists("generated_complexes.csv"):

                try:
                    df = pd.read_csv("generated_complexes.csv")
                except:
                    time.sleep(1)
                    continue

                if not df.empty:

                    for c in df.columns:
                        if "zfs" in c.lower():
                            zfs_col = c
                            break

                    if zfs_col is not None:
                        break

            time.sleep(1)

        if df is None or zfs_col is None:
            st.warning("Prediction column still not ready")
            continue

        # ================= FIND BEST COMPLEX =================

        df["error"] = abs(df[zfs_col] - target_zfs)

        df = df.sort_values("error")

        best = df.iloc[0]

        st.success(f"Best ZFS so far: {best[zfs_col]:.2f}")

        best_table = pd.DataFrame([{
            "Ligands": best.get("ligands", "N/A"),
            "Donor Pattern": best.get("donor_pattern", "N/A"),
            "CN": best.get("CN", "N/A"),
            "Predicted ZFS": best[zfs_col],
            "E/D": best.get("ed_pred", best.get("E_D", "N/A")),
            "Error": best["error"]
        }])

        st.dataframe(best_table)

        # ================= STOP IF TARGET ACHIEVED =================

        if best[zfs_col] <= target_zfs:

            st.success("🎯 Target achieved")

            upload_pipeline_to_drive(target_zfs, mode)

            break

        # ================= SAVE STATE =================

        if any(os.path.exists(f) for f in [
            "mutated_ligands.csv",
            "mutation_lineage.csv",
            "generated_complexes.csv",
            "elite_parents.csv",
        ]):

            upload_pipeline_to_drive(target_zfs, mode)

            st.write("☁️ GA state saved")

    # ================= FINAL DISPLAY =================

    st.subheader("🏆 Elite ligand combinations")

    if os.path.exists("elite_parents.csv"):

        elite = pd.read_csv("elite_parents.csv")

        if not elite.empty:
            st.dataframe(elite)
