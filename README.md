# ZFS-driven Ligand Design App

Streamlit app for generating ligand SMILES combinations
targeting a desired ZFS value using a GA + GNN oracle.

## Run locally

```bash
git clone https://github.com/<your-username>/zfs-ligand-app.git
cd zfs-ligand-app
pip install -r requirements.txt
streamlit run app.py


CCDC addition: CCDC_lookup.csv is metadata sourced from the CCDC-enabled dataset. The original GA.csv remains unchanged so generation behavior is preserved.
