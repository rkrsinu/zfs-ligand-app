import streamlit as st
import subprocess
import pandas as pd
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable   # 🔥 critical for Streamlit

st.set_page_config(page_title="ZFS-driven Ligand Design", layout="wide")
st.title("🔬 ZFS-driven Ligand SMILES Generator")

# ---------------- SIDEBAR ----------------

target_zfs = st.sidebar.number_input("Target ZFS", value=-180.0, step=5.0)
mode = st.sidebar.selectbox("Mode", ["optimized", "crystal"])
max_generations = st.sidebar.number_input("Max GA generations", value=200, step=50)

run_button = st.sidebar.button("🚀 Run")

# ---------------- ELITE DISPLAY ----------------

def show_elite():
    path = os.path.join(BASE_DIR, "elite_parents.csv")
    if not os.path.exists(path):
        st.warning("No elite results found yet.")
        return

    df = pd.read_csv(path)

    st.subheader("🏆 Elite ligand combinations")
    st.dataframe(df[["ligands", "zfs_pred", "ed_pred", "donor_sum"]])

    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "elite_parents.csv")

# ---------------- RUN ----------------

if run_button:

    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    st.info("Checking database…")

    ret = subprocess.call([PYTHON, "00_target_decision.py", str(target_zfs)])

    retrieved_path = os.path.join(BASE_DIR, "retrieved_solution.csv")

    if ret == 0 and os.path.exists(retrieved_path):
        st.success("🎯 Direct database match found (±10 cm⁻¹)")
        st.dataframe(pd.read_csv(retrieved_path))
        st.stop()

    st.warning("No DB hit → starting GA")

    progress = st.progress(0)
    status = st.empty()

    for gen in range(1, int(max_generations) + 1):

        status.text(f"Generation {gen}")
        os.environ["GA_GEN"] = str(gen)

        subprocess.call([PYTHON, "03_ligand_mutation.py"])
        subprocess.call([PYTHON, "04_build_complexes.py"])
        subprocess.call([PYTHON, "05_oracle_screen.py"])

        progress.progress(gen / max_generations)

        elite_path = os.path.join(BASE_DIR, "elite_parents.csv")

        if os.path.exists(elite_path):
            elite = pd.read_csv(elite_path)
            best = elite["zfs_pred"].min()

            st.write(f"Best ZFS so far: **{best:.2f}**")

            if best <= target_zfs:
                st.success(f"🎯 Target achieved at generation {gen}")
                show_elite()
                break

        time.sleep(0.1)

    else:
        st.warning("Stopped before reaching target")
        show_elite()
