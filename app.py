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


def show_mutation_details(row):
    """Show exactly which generated ligand came from which parent/CCDC and mutation."""
    ligands = parse_list(row.get("ligands", "")) if ";" not in str(row.get("ligands", "")) else [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    parents = parse_list(row.get("parent_ligands", ""))
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))

    n = max(len(ligands), len(parents), len(ccdcs), len(muts))
    records = []
    for i in range(n):
        child = ligands[i] if i < len(ligands) else ""
        parent = parents[i] if i < len(parents) else ""
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else ""
        records.append({
            "Ligand": f"L{i+1}",
            "Generated ligand": child,
            "Parent ligand": parent,
            "Parent CCDC": ccdc,
            "Generation / operation": mutation,
        })

    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def show_complex_assembly(row):
    """Give an experimentalist-friendly coordination blueprint.

    A SMILES combination does not uniquely define 3-D orientation, so this
    section deliberately reports the donor-slot blueprint rather than inventing
    Cartesian coordinates.
    """
    donor_values = parse_list(row.get("donor_list", ""))
    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]

    st.markdown("### 🧩 Complex construction & orientation")
    st.info(
        "The GA generates the ligand composition and donor count, but SMILES alone "
        "does not determine a unique 3-D coordination geometry. The blueprint below "
        "shows how the six donor sites are to be occupied; the final orientation and "
        "bond lengths/angles should be obtained from an experimentally known structure "
        "or a geometry optimization."
    )

    rows = []
    for i, (lig, d) in enumerate(zip(ligands, donor_values), 1):
        try:
            d_int = int(float(d))
        except Exception:
            d_int = d
        rows.append({
            "Ligand": f"L{i}",
            "SMILES": lig,
            "Donor atoms": d_int,
            "Coordination role": "Chelating ligand" if isinstance(d_int, int) and d_int > 1 else "Monodentate ligand",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    donor_total = sum(int(float(x)) for x in donor_values) if donor_values else 0
    st.markdown(f"**Total donor sites:** {donor_total} / 6")

    if donor_values:
        st.markdown("**Octahedral donor-slot concept:**")
        st.code("""                    Axial donor
                         │
              Equatorial donor ─ Co ─ Equatorial donor
                       /       \
              Equatorial       Equatorial
                         │
                    Axial donor

Assign the six donor atoms from the listed ligands to the six coordination sites.
For multidentate ligands, adjacent sites should be used according to their
chelate topology; do not infer an exact orientation from the SMILES alone.""")


def show_ligand_structures(row):
    """Render parent -> generated 2-D ligand structures."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception:
        st.warning("RDKit is not available in this Streamlit environment, so 2-D structures cannot be rendered.")
        return

    ligands = [x.strip() for x in str(row.get("ligands", "")).split(";") if x.strip()]
    parents = parse_list(row.get("parent_ligands", ""))
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    muts = parse_list(row.get("mutations", ""))

    st.markdown("### 🧬 Parent → generated ligand structures")
    for i, child_smi in enumerate(ligands):
        parent_smi = parents[i] if i < len(parents) else ""
        ccdc = ccdcs[i] if i < len(ccdcs) else ""
        mutation = muts[i] if i < len(muts) else "database_ligand"

        child_mol = Chem.MolFromSmiles(child_smi)
        parent_mol = Chem.MolFromSmiles(parent_smi) if parent_smi else None

        st.markdown(f"**L{i+1} — {mutation} — Parent CCDC: {ccdc or 'not found'}**")
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            if parent_mol is not None:
                st.image(Draw.MolToImage(parent_mol, size=(450, 320)), caption="Parent ligand")
            else:
                st.warning("Parent structure unavailable")
        with col2:
            st.markdown("<div style='text-align:center; font-size:40px; padding-top:110px;'>→</div>", unsafe_allow_html=True)
        with col3:
            if child_mol is not None:
                st.image(Draw.MolToImage(child_mol, size=(450, 320)), caption="Generated ligand")
            else:
                st.warning(f"RDKit could not draw generated SMILES: {child_smi}")

        st.caption(f"Mutation operation: {mutation} | Parent CCDC: {ccdc or 'not found'}")
        st.code(f"Parent:  {parent_smi}\nChild:   {child_smi}")

    st.info(
        "This parent → child diagram shows the exact structural transformation represented "
        "by the GA. It is a molecular-design representation, not a validated laboratory "
        "synthetic procedure. The actual reaction conditions/reagents must be established "
        "from chemical literature and experimental validation."
    )


def show_experimental_lookup(row):
    """Show experimental lookup once, after the GA campaign rather than every generation."""
    ccdcs = parse_list(row.get("parent_ccdcs", ""))
    ccdcs = [c for c in ccdcs if c and c.lower() != "nan"]
    unique_ccdcs = list(dict.fromkeys(ccdcs))

    st.markdown("### 🧪 Experimental procedure lookup")
    st.write(
        "The **Parent CCDC(s)** identify known crystal structures from which the "
        "starting ligands were taken. Use these CCDC numbers to locate the original "
        "crystal-structure paper/supporting information and reported synthetic procedure."
    )
    if unique_ccdcs:
        st.markdown("**Parent CCDC(s):** " + ", ".join(unique_ccdcs))
    else:
        st.warning("No parent CCDC was resolved for this candidate.")

    st.caption(
        "Important: the CCDC provides provenance for the parent ligand/complex. "
        "It does not by itself prove that the newly mutated ligand has the same "
        "synthetic procedure. The new ligand must be synthesized/validated separately."
    )


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
            st.markdown("### 🧪 Experimental procedure lookup")
            st.write("Use the CCDC number above to locate the corresponding crystal-structure paper, supporting information, and reported synthesis.")
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
                    "Parent CCDC(s)": best_row.get("parent_CCDC_for_experiment", best_row.get("parent_ccdcs", "")),
                    "Donor Pattern": donor_list,
                    "Total Donors": donor_sum,
                    "Predicted D": D_value,
                    "E/D": ED_value,
                }])
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Requested: ligand-specific mutation + CCDC mapping.
                st.markdown("### 🧬 Ligand mutation & parent CCDC")
                show_mutation_details(best_row)

                # Requested: how the six-coordinate complex is assembled.
                show_complex_assembly(best_row)

                # Visual representation of each generated ligand.
                show_ligand_structures(best_row)

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
    # FINAL EXPERIMENTAL INFORMATION — SHOWN ONLY ONCE
    # ======================================================
    if final_best is not None:
        st.markdown("---")
        st.markdown("# 🧪 Experimental information for the final selected candidate")
        show_experimental_lookup(final_best)

        st.markdown("### 📋 Final candidate summary")
        final_cols = {
            "Generated ligand(s)": final_best.get("ligands", ""),
            "Parent ligand(s)": final_best.get("parent_ligands", ""),
            "Parent CCDC(s)": final_best.get("parent_CCDC_for_experiment", final_best.get("parent_ccdcs", "")),
            "Mutation(s)": final_best.get("mutations", ""),
            "Donor pattern": final_best.get("donor_list", ""),
            "Predicted D": final_best.get("zfs_pred", ""),
            "E/D": final_best.get("ed_pred", ""),
        }
        st.dataframe(pd.DataFrame([final_cols]), use_container_width=True, hide_index=True)

        st.markdown("### ⚗️ What can and cannot be shown as a synthesis")
        st.write(
            "The application can show the parent ligand structures, the exact mutation "
            "operation used by the GA, the resulting SMILES/2-D structures, the six-donor "
            "coordination blueprint, and the parent CCDC provenance. A reliable step-by-step "
            "synthetic procedure for a newly generated ligand cannot be inferred from SMILES "
            "alone. When a parent CCDC has a reported synthesis, that literature procedure "
            "can be used as the starting experimental reference, followed by chemical validation "
            "of the proposed mutation."
        )
