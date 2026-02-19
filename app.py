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

target_zfs = st.sidebar.number_input("Target ZFS", value=-180.0)
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
        st.success("Direct database hit")
        df = pd.read_csv("retrieved_solution.csv")
        st.dataframe(df)
        st.stop()

    st.warning("No DB hit → starting GA")

    # ===== RESTORE PREVIOUS STATE =====

    try:
        download_pipeline_from_drive(target_zfs, mode)
        st.success("Previous GA state restored")
    except Exception as e:
        st.warning(f"No previous state found: {e}")

    progress = st.progress(0)

    for gen in range(1, int(max_gen) + 1):

        progress.progress(gen / max_gen)
        st.subheader(f"Generation {gen}")

        # ===== FIRST RUN INIT =====
        if gen == 1 and not os.path.exists("elite_parents.csv"):

            subprocess.call([PYTHON, "00_build_ligand_donor_map.py"])
            subprocess.call([PYTHON, "01_select_seeds.py"])
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
            st.warning("elite_parents.csv not created")

        # ===== SAVE STATE TO DRIVE =====
        try:
            upload_pipeline_to_drive(target_zfs, mode)
            st.write("☁️ State saved to Drive")
        except Exception as e:
            st.error(f"Drive upload failed: {e}")

    # ===== FINAL DISPLAY =====

    st.subheader("🏆 Elite ligand combinations")

    if os.path.exists("elite_parents.csv"):
        st.dataframe(pd.read_csv("elite_parents.csv"))
    else:
        st.warning("No elite results found")
