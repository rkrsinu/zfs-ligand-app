import streamlit as st
import os
import subprocess
import pandas as pd

from gdrive_save import (
    prepare_drive_folders,
    upload_generation_outputs,
    upload_elite,
    download_previous_state,
)

st.set_page_config(layout="wide")

st.title("🔬 ZFS-driven Ligand SMILES Generator")

target_zfs = st.number_input("Target ZFS", value=-180.0)
mode = st.selectbox("Mode", ["crystal", "optimized"])
max_gen = st.number_input("Max GA generations", value=5)

if st.button("🚀 Run"):

    st.info("Checking database...")

    previous = download_previous_state(target_zfs, mode)

    if previous:
        st.success("Previous state found → continuing GA")
    else:
        st.info("No previous state found → fresh GA run")

    for gen in range(1, int(max_gen) + 1):

        st.subheader(f"Generation {gen}")

        if gen == 1 and not previous:
            st.write("Building donor map")
            subprocess.run(["python", "00_build_ligand_donor_map.py"])

            st.write("Selecting seed complexes")
            subprocess.run(["python", "01_select_seeds.py"])

            st.write("Extracting seed ligands")
            subprocess.run(["python", "02_extract_seed_ligands.py"])

        subprocess.run(["python", "03_ligand_mutation.py"])
        subprocess.run(["python", "04_build_complexes.py"])
        subprocess.run(["python", "05_oracle_screen.py", str(target_zfs), mode])

        upload_generation_outputs(target_zfs, mode)

        if os.path.exists("elite_parents.csv"):
            st.success("Elite found in this generation")
            upload_elite(target_zfs, mode)
            break
        else:
            st.warning("elite_parents.csv not created in this generation")

    if os.path.exists("elite_parents.csv"):
        df = pd.read_csv("elite_parents.csv")
        st.dataframe(df)
    else:
        st.warning("No elite results found yet.")
