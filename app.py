import streamlit as st
import pandas as pd

from engine.database_lookup import load_database
from engine.target_decision import decide_from_database

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="ZFS-driven Ligand SMILES Generator",
    layout="wide"
)

st.title("🔬 ZFS-driven Ligand SMILES Generator")
st.caption(
    "Database-first ZFS lookup → GA fallback (offline only)"
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("🎯 Target settings")

target_zfs = st.sidebar.number_input(
    "Target ZFS (cm⁻¹)",
    min_value=-500.0,
    max_value=500.0,
    value=-160.0,
    step=1.0,
    format="%.2f"
)

st.sidebar.markdown(f"**Using Target:** `{target_zfs:.2f} cm⁻¹`")

mode = st.sidebar.selectbox(
    "Mode",
    ["crystal", "optimized"]
)

max_gen = st.sidebar.number_input(
    "Max GA generations (offline)",
    min_value=1,
    max_value=500,
    value=20
)

run = st.sidebar.button("🚀 Run Search")

# ---------------- MAIN LOGIC ---------------- #

if run:
    st.info("Starting pipeline...")

    try:
        df, zfs_col = load_database(mode)

        st.success(f"Database loaded ({len(df)} entries)")
        st.write(
            f"ZFS range: {df[zfs_col].min():.2f} → {df[zfs_col].max():.2f} cm⁻¹"
        )

        found, hits = decide_from_database(
            df=df,
            zfs_col=zfs_col,
            target_zfs=target_zfs,
            tol=1.0
        )

        if found:
            st.success("✅ Target found in database — GA skipped")

            st.subheader("Best matching complexes")

            display_cols = [
                c for c in hits.columns if c != "delta"
            ]

            st.dataframe(
                hits[display_cols].head(10),
                use_container_width=True
            )

        else:
            st.warning("⚠️ No database match → GA required")
            st.progress(100)
            st.write(f"Reached generation {max_gen}")
            st.warning("Stopped before reaching target")
            st.info("No elite results found yet.")

    except Exception as e:
        st.error("❌ Error during execution")
        st.exception(e)

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.caption(
    "ZFS-driven ligand design · Database-first architecture · Streamlit UI"
)
