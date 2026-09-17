import streamlit as st
import subprocess
import sys
import os
import ast
import html
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
mode = st.sidebar.selectbox("Mode", ["X-ray", "DFT"])
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

        # ================= CCDC DISPLAY FOR DIRECT DATABASE HIT =================
        # Original database result is kept intact; only CCDC/provenance display is added.
        _ccdc_lookup = {}
        if os.path.exists("CCDC_lookup.csv"):
            try:
                _cdf = pd.read_csv("CCDC_lookup.csv")
                if "FileName" in _cdf.columns and "CCDC" in _cdf.columns:
                    for _, _rr in _cdf.iterrows():
                        _key = str(_rr["FileName"]).strip()
                        _val = str(_rr["CCDC"]).strip()
                        if _key and _key.lower() != "nan" and _val and _val.lower() != "nan":
                            _ccdc_lookup[_key] = _val
            except Exception:
                pass

        _display = result.copy()
        if "FileName" in _display.columns:
            _display["CCDC"] = _display["FileName"].astype(str).str.strip().map(_ccdc_lookup).fillna("")
            _cols = list(_display.columns)
            _cols.remove("CCDC")
            _cols.insert(_cols.index("FileName") + 1, "CCDC")
            _display = _display[_cols]

        st.dataframe(_display)

        # Same compact provenance style used below GA generations.
        if not result.empty:
            _r = result.iloc[0]
            _file = str(_r.get("FileName", "")).strip()
            _ccdc = _ccdc_lookup.get(_file, "")
            st.markdown("### 🧬 Ligand provenance")

            _cards = []
            for _i in range(1, 7):
                _col = f"L{_i}"
                if _col not in result.columns:
                    continue
                _lig = str(_r.get(_col, "")).strip()
                if not _lig or _lig.upper() == "X":
                    continue
                _cards.append(f"""
                <div style="border:1px solid #14532d; border-radius:10px; padding:12px 14px; margin:8px 0; background:#052e16;">
                  <div style="font-size:16px; font-weight:700; color:#4ade80;">L{_i} · REPORTED LIGAND</div>
                  <div style="margin-top:6px; color:#e5e7eb; word-break:break-all;"><b>Ligand:</b> {html.escape(_lig)}</div>
                  <div style="margin-top:6px; color:#86efac;"><b>CCDC:</b> {html.escape(_ccdc) if _ccdc else "Not available"}</div>
                </div>
                """)

            if _cards:
                st.markdown("".join(_cards), unsafe_allow_html=True)

        st.stop()

    st.warning("⚠️ No suitable database hit found → 🚀 Entering AI-guided design mode")

    # ================= RESTORE =================

    restored = download_pipeline_from_drive(target_zfs, mode)

    if restored:
        st.success("♻️ Resuming inverse molecular design")
        first_run = False
    else:
        st.info("🆕 Initiating inverse molecular design")
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

        # ================= SHOW BEST RESULT =================

        if os.path.exists("elite_parents.csv"):

            elite = pd.read_csv("elite_parents.csv")

            if not elite.empty:

                best_row = elite.sort_values("zfs_pred").iloc[0]

                ligand_combo = best_row["ligands"]
                donor_list = best_row["donor_list"]
                donor_sum = best_row["donor_sum"]
                D_value = best_row["zfs_pred"]
                ED_value = best_row["ed_pred"]
                CCDC_value = best_row.get("parent_CCDC_for_experiment", "")

                st.success(f"Best ZFS so far: {D_value:.2f}")

                result_df = pd.DataFrame([{
                    "Ligand Combination": ligand_combo,
                    "CCDC": CCDC_value,
                    "Donor Pattern": donor_list,
                    "Total Donors": donor_sum,
                    "Predicted D": D_value,
                    "E/D": ED_value
                }])

                st.dataframe(result_df)

                # ================= LIGAND PROVENANCE =================
                # Added only to show whether each ligand is reported or modified,
                # and, for modified ligands, the modification and immediate parent.
                def _parse_meta_list(value):
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        return []
                    try:
                        parsed = ast.literal_eval(str(value))
                        if isinstance(parsed, (list, tuple)):
                            return [str(x).strip() for x in parsed]
                    except Exception:
                        pass
                    return [x.strip() for x in str(value).split(';') if x.strip()]

                def _mutation_label(mutation):
                    labels = {
                        "methyl_addition": "Methyl addition",
                        "ethyl_addition": "Ethyl addition",
                        "isopropyl_addition": "Isopropyl addition",
                        "atom_type_substitution": "Atom-type substitution",
                        "halogen_exchange": "Halogen exchange",
                    }
                    return labels.get(str(mutation).strip(), str(mutation).replace('_', ' ').title())

                def _mutation_style(mutation):
                    styles = {
                        "methyl_addition": ("#7c3aed", "#f5f3ff"),
                        "ethyl_addition": ("#2563eb", "#eff6ff"),
                        "isopropyl_addition": ("#0891b2", "#ecfeff"),
                        "atom_type_substitution": ("#d97706", "#fffbeb"),
                        "halogen_exchange": ("#dc2626", "#fef2f2"),
                    }
                    return styles.get(str(mutation).strip(), ("#475569", "#f8fafc"))

                def _resolve_provenance(ligand, parent_ligand, mutation, parent_ccdc):
                    mutation = str(mutation).strip()
                    parent_ligand = str(parent_ligand).strip()
                    parent_ccdc = str(parent_ccdc).strip()

                    if mutation not in {"", "database_ligand", "elite_parent"}:
                        return mutation, parent_ligand, parent_ccdc

                    # An elite ligand can itself be a previously modified ligand.
                    # Recover its immediate mutation/parent/CCDC from the stored lineage.
                    if os.path.exists("mutation_lineage.csv"):
                        try:
                            lineage_df = pd.read_csv("mutation_lineage.csv")
                            matches = lineage_df[lineage_df["child"].astype(str).str.strip() == str(ligand).strip()]
                            if not matches.empty:
                                r = matches.iloc[-1]
                                m = str(r.get("mutation", "")).strip()
                                p = str(r.get("parent", "")).strip()
                                c = str(r.get("parent_ccdc", "")).strip()
                                if m and m != "database_ligand":
                                    return m, p, c or parent_ccdc
                        except Exception:
                            pass

                    return "database_ligand", ligand, parent_ccdc

                ligand_items = _parse_meta_list(ligand_combo)
                parent_items = _parse_meta_list(best_row.get("parent_ligands", ""))
                parent_ccdc_items = _parse_meta_list(best_row.get("parent_ccdcs", ""))
                mutation_items = _parse_meta_list(best_row.get("mutations", ""))

                st.markdown("### 🧬 Ligand provenance")

                cards = []
                for i, ligand in enumerate(ligand_items):
                    parent = parent_items[i] if i < len(parent_items) else ligand
                    parent_ccdc = parent_ccdc_items[i] if i < len(parent_ccdc_items) else ""
                    mutation = mutation_items[i] if i < len(mutation_items) else "database_ligand"
                    mutation, parent, parent_ccdc = _resolve_provenance(ligand, parent, mutation, parent_ccdc)

                    if mutation == "database_ligand":
                        cards.append(f"""
                        <div style="border:1px solid #14532d; border-radius:10px; padding:12px 14px; margin:8px 0; background:#052e16;">
                          <div style="font-size:16px; font-weight:700; color:#4ade80;">L{i+1} · REPORTED LIGAND</div>
                          <div style="margin-top:6px; color:#e5e7eb; word-break:break-all;"><b>Ligand:</b> {html.escape(str(ligand))}</div>
                          <div style="margin-top:6px; color:#86efac;"><b>CCDC:</b> {html.escape(parent_ccdc) if parent_ccdc else "Not available"}</div>
                        </div>
                        """)
                    else:
                        fg, bg = _mutation_style(mutation)
                        label = _mutation_label(mutation)
                        cards.append(f"""
                        <div style="border:1px solid {fg}; border-radius:10px; padding:12px 14px; margin:8px 0; background:#111827;">
                          <div style="font-size:16px; font-weight:700; color:#60a5fa;">L{i+1} · MODIFIED LIGAND</div>
                          <div style="margin-top:7px;">
                            <span style="display:inline-block; padding:4px 9px; border-radius:999px; background:{bg}; color:{fg}; font-weight:700; font-size:13px;">{html.escape(label)}</span>
                          </div>
                          <div style="margin-top:8px; color:#e5e7eb; word-break:break-all;"><b>Modified ligand:</b> {html.escape(str(ligand))}</div>
                          <div style="margin-top:6px; color:#cbd5e1; word-break:break-all;"><b>Parent ligand:</b> {html.escape(str(parent))}</div>
                          <div style="margin-top:6px; color:#c4b5fd;"><b>Parent CCDC:</b> {html.escape(parent_ccdc) if parent_ccdc else "Not available"}</div>
                        </div>
                        """)

                if cards:
                    st.markdown("".join(cards), unsafe_allow_html=True)

                # stop if target achieved
                if D_value <= target_zfs:
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
            st.write("☁️ Design campaign checkpoint saved")
