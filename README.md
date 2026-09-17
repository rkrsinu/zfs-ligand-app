# ZFS-driven Ligand SMILES Generator

A Streamlit interface for inverse ligand design using a genetic algorithm (GA) and GNN ZFS/E/D oracles.

## Important architecture

The Streamlit process is **not** the long-running GA engine anymore.

```text
Streamlit UI
    |
    +--> ga_worker.py  --------> GA generations
              |
              +--> ligand mutation
              +--> complex generation
              +--> GNN oracle (models loaded ONCE)
              +--> local checkpoint after every generation
              +--> Google Drive backup every N generations
```

This prevents the previous design where one Streamlit execution had to remain alive for hundreds of generations.

## Resume behavior

The checkpoint stores the internal GA generation. For example:

```text
Run 1: visible Generation 1 ... 18
        internal Generation 1 ... 18
        checkpoint -> next_generation = 19

Streamlit/container interruption

Run 2: visible Generation 1 ...
        internal Generation 19 ...
```

Thus the **visible counter resets to Generation 1 for every Run click**, while the actual GA campaign resumes from its saved internal generation.

## Google Drive strategy

Only the minimum resume state is synchronized:

- `ga_checkpoint.csv`
- `elite_parents.csv`

The donor map can be rebuilt from `GA.csv` / `opt_D.csv` after a container restart, so large per-generation files do not need to be uploaded every generation.

Default backup frequency: every 5 generations, plus at target achievement and at the end of a Run.

Change this in the sidebar.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For command-line execution:

```bash
python 06_run_until_target.py -180 crystal 500
```

or

```bash
python 06_run_until_target.py -180 optimized 500
```

## Streamlit secrets

Create `.streamlit/secrets.toml` with:

```toml
GDRIVE_REFRESH_TOKEN = "..."
GDRIVE_CLIENT_ID = "..."
GDRIVE_CLIENT_SECRET = "..."
GDRIVE_FOLDER_ID = "..."
```

Never commit the real credentials or service-account JSON files.

## Main files

- `app.py` — Streamlit UI and live monitor
- `ga_worker.py` — persistent GA worker
- `oracle_engine.py` — reusable GNN oracle; loads models once
- `gdrive_save.py` — lightweight checkpoint backup/restore
- `00_build_ligand_donor_map.py` — donor map
- `01_select_seeds.py` — seed selection
- `02_extract_seed_ligands.py` — seed ligand extraction
- `03_ligand_mutation.py` — reaction-based mutation + provenance
- `04_build_complexes.py` — six-donor complex construction
- `05_oracle_screen.py` — stand-alone oracle wrapper
- `06_run_until_target.py` — CLI wrapper

## Result provenance shown in the UI

For each ligand the result table shows only:

- `L1 · Reported ligand · CCDC XXXXXXX`
- `L2 · Mutated from reported ligand · CCDC XXXXXXX`

The detailed parent SMILES/mutation history remains in the CSV lineage files but is not displayed in the main UI.
