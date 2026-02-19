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
mode = st.sidebar.selectbox("Mode", ["crystal", "optimized"])
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
        df = pd.read_csv("retrieved_solution.csv")
        st.dataframe(df)
        st.stop()

    st.warning("No valid database match → starting GA")

    # ================= RESTORE STATE =================

    first_run = True

    try:
        restored = download_pipeline_from_drive(target_zfs, mode)

        if restored:
            st.success("Previous GA state restored from Drive")
            first_run = False
        else:
            st.info("No previous state found → fresh GA run")

    except Exception as e:
        st.warning(f"Drive restore skipped: {e}")

    progress = st.progress(0)

    # ================= GA LOOP =================

    for gen in range(1, int(max_gen) + 1):

        progress.progress(gen / max_gen)
        st.subheader(f"Generation {gen}")

        # ===== FIRST RUN INITIALIZATION =====
        if gen == 1 and first_run:

            st.write("Building donor map")
            subprocess.call([PYTHON, "00_build_ligand_donor_map.py"])

            st.write("Selecting seed complexes")
            subprocess.call([PYTHON, "01_select_seeds.py"])

            st.write("Extracting seed ligands")
            subprocess.call([PYTHON, "02_extract_seed_ligands.py"])

        # ===== GA CORE =====

        subprocess.call([PYTHON, "03_ligand_mutation.py"])
        subprocess.call([PYTHON, "04_build_complexes.py"])
        subprocess.call([PYTHON, "05_oracle_screen.py"])

        # ===== SHOW BEST RESULT =====

        if os.path.exists("elite_parents.csv"):

            elite = pd.read_csv("elite_parents.csv")

            if not elite.empty:

                best = elite["zfs_pred"].min()
                st.success(f"Best ZFS so far: {best:.2f}")

                if best <= target_zfs:
                    st.success("🎯 Target achieved")
                    break

        else:
            st.warning("elite_parents.csv not created in this generation")

        # ===== SAVE STATE TO DRIVE =====

        try:
            upload_pipeline_to_drive(target_zfs, mode)
            st.write("☁️ GA state saved to Drive")

        except Exception as e:
            st.error(f"Drive upload failed: {e}")

    # ================= FINAL DISPLAY =================

    st.subheader("🏆 Elite ligand combinations")

    if os.path.exists("elite_parents.csv"):

        elite = pd.read_csv("elite_parents.csv")

        if not elite.empty:
            st.dataframe(elite)
        else:
            st.warning("Elite file exists but empty")

    else:
        st.warning("No elite results found")
