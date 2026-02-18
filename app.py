import streamlit as st
import subprocess
import pandas as pd
import os
import sys
import time

# ✅ Google Drive saver
from gdrive_save import save_results_to_drive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

st.set_page_config(page_title="ZFS-driven Ligand Design", layout="wide")

st.title("🔬 ZFS-driven Ligand SMILES Generator")

st.markdown(
    "This app uses a **GA + GNN oracle pipeline** to generate ligand combinations for a target **ZFS value**."
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("🎯 Target settings")

target_zfs = st.sidebar.number_input("Target ZFS (cm⁻¹)", value=-180.0, step=5.0)

mode = st.sidebar.selectbox("Mode", ["optimized", "crystal"])

max_generations = st.sidebar.number_input("Max GA generations", value=200, step=50)

run_button = st.sidebar.button("🚀 Run")


# ----------------------------------------------------------
# FILTER: remove FileNames starting with 'a'
# ----------------------------------------------------------

def remove_a_filenames(df):
    if "FileName" in df.columns:
        df = df[~df["FileName"].astype(str).str.startswith("a")]
    return df


# ----------------------------------------------------------
# ELITE DISPLAY
# ----------------------------------------------------------

def show_elite():

    elite_path = os.path.join(BASE_DIR, "elite_parents.csv")

    if not os.path.exists(elite_path):
        st.warning("No elite results found yet.")
        return

    df = pd.read_csv(elite_path)
    df = remove_a_filenames(df)

    st.subheader("🏆 Elite ligand combinations")

    show_cols = [c for c in ["FileName", "ligands", "zfs_pred", "ed_pred", "donor_sum"] if c in df.columns]

    st.dataframe(df[show_cols], use_container_width=True)

    st.download_button(
        "⬇️ Download elite CSV",
        df.to_csv(index=False),
        "elite_parents.csv",
        mime="text/csv",
    )


# ----------------------------------------------------------
# SAFE GOOGLE DRIVE SAVE
# ----------------------------------------------------------

def save_to_drive_safe():
    try:
        save_results_to_drive(target_zfs, mode)
        st.info("☁️ Results saved to Google Drive")
    except Exception as e:
        st.warning("Drive upload skipped (not configured yet)")


# ----------------------------------------------------------
# RUN
# ----------------------------------------------------------

if run_button:

    st.info("Checking database…")

    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    ret = subprocess.call([PYTHON, "00_target_decision.py", str(target_zfs)])

    retrieved_path = os.path.join(BASE_DIR, "retrieved_solution.csv")

    # ---------------- DATABASE HIT ----------------

    if ret == 0 and os.path.exists(retrieved_path):

        st.success("🎯 Direct database match found (±10 cm⁻¹)")

        df = pd.read_csv(retrieved_path)
        df = remove_a_filenames(df)

        if len(df) == 0:
            st.warning("Only 'a' filenames were found → starting GA instead")
        else:
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Download database hits",
                df.to_csv(index=False),
                "retrieved_solution.csv",
                mime="text/csv",
            )

            st.stop()

    # ---------------- GA START ----------------

    st.warning("No valid database match → starting GA")

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
            elite = remove_a_filenames(elite)

            if len(elite) == 0:
                continue

            best = elite["zfs_pred"].min()

            st.write(f"Best ZFS so far: **{best:.2f}**")

            if best <= target_zfs:

                st.success(f"🎯 Target achieved at generation {gen}")

                save_to_drive_safe()   # ☁️ SAVE TO DRIVE

                show_elite()
                break

        time.sleep(0.1)

    else:
        st.warning("Stopped before reaching target")

        save_to_drive_safe()   # ☁️ SAVE TO DRIVE

        show_elite()


st.markdown("---")
st.markdown("**ZFS-driven ligand design · GA + GNN oracle · Streamlit interface**")
