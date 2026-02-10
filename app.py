# ==========================================================
# app.py
# Streamlit interface for ZFS-driven ligand generation
# ==========================================================

import streamlit as st
import subprocess
import pandas as pd
import os
import time

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------

st.set_page_config(
    page_title="ZFS-driven Ligand Design",
    layout="wide"
)

st.title("🔬 ZFS-driven Ligand SMILES Generator")

st.markdown("""
This app uses an existing **GA + GNN oracle pipeline**  
to generate **ligand SMILES combinations** for a target **ZFS value**.
""")

# ----------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------

st.sidebar.header("🎯 Target settings")

target_zfs = st.sidebar.number_input(
    "Target ZFS (cm⁻¹)",
    value=-180.0,
    step=5.0
)

mode = st.sidebar.selectbox(
    "Mode",
    ["optimized", "crystal"]
)

max_generations = st.sidebar.number_input(
    "Max GA generations",
    value=200,
    step=50
)

run_button = st.sidebar.button("🚀 Run GA Search")

# ----------------------------------------------------------
# Helper: display elite results
# ----------------------------------------------------------

def show_elite():
    if not os.path.exists("elite_parents.csv"):
        st.warning("No elite results found yet.")
        return

    df = pd.read_csv("elite_parents.csv")

    st.subheader("🏆 Elite ligand combinations")

    show_cols = []
    for c in ["ligands", "zfs_pred", "ed_pred", "donor_sum"]:
        if c in df.columns:
            show_cols.append(c)

    st.dataframe(df[show_cols], use_container_width=True)

    st.download_button(
        label="⬇️ Download elite CSV",
        data=df.to_csv(index=False),
        file_name="elite_parents.csv",
        mime="text/csv"
    )

# ----------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------

if run_button:

    st.info("Starting GA pipeline…")

    # ------------------------------------------------------
    # Set environment variables
    # ------------------------------------------------------

    os.environ["MODE"] = mode
    os.environ["TARGET_ZFS"] = str(target_zfs)

    # ------------------------------------------------------
    # First: database lookup
    # ------------------------------------------------------

    with st.spinner("Checking database…"):
        ret = subprocess.call(
            ["python", "00_target_decision.py", str(target_zfs)]
        )

    if ret == 0 and os.path.exists("retrieved_solution.csv"):
        st.success("🎯 Solution found in database")

        df = pd.read_csv("retrieved_solution.csv")
        st.dataframe(df, use_container_width=True)
        st.stop()

    st.warning("No database match → starting GA")

    progress = st.progress(0.0)
    status = st.empty()

    # ------------------------------------------------------
    # GA loop
    # ------------------------------------------------------

    for gen in range(1, int(max_generations) + 1):

        status.text(f"Generation {gen}")

        os.environ["GA_GEN"] = str(gen)

        subprocess.call(["python", "03_ligand_mutation.py"])
        subprocess.call(["python", "04_build_complexes.py"])
        subprocess.call(["python", "05_oracle_screen.py"])

        progress.progress(gen / max_generations)

        # Check elite
        if os.path.exists("elite_parents.csv"):
            elite = pd.read_csv("elite_parents.csv")
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

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.markdown("---")
st.markdown(
    "**ZFS-driven ligand design** · GA + GNN oracle · Streamlit interface"
)
