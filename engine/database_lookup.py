import pandas as pd
import os

DATA_DIR = "data"

def load_database(mode: str):
    """
    Load GA.csv and return dataframe + correct ZFS column
    """

    path = os.path.join(DATA_DIR, "GA.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("GA.csv not found in data/")

    df = pd.read_csv(path)

    if mode == "crystal":
        zfs_col = "zfs"
    else:
        zfs_col = "zfs_opt"

    if zfs_col not in df.columns:
        raise ValueError(f"Column '{zfs_col}' missing in GA.csv")

    return df, zfs_col
