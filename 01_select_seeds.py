import pandas as pd
import os

# ----------------------------------------------------------
# Detect mode from Streamlit
# ----------------------------------------------------------

mode = os.getenv("MODE", "crystal")

# ----------------------------------------------------------
# Load correct database
# ----------------------------------------------------------

if mode == "optimized":

    df = pd.read_csv("opt_D.csv")
    zfs_col = "opt_zfs"

else:  # crystal

    df = pd.read_csv("GA.csv")
    zfs_col = "zfs"

print("MODE =", mode)
print("Database loaded:", "opt_D.csv" if mode=="optimized" else "GA.csv")

# ----------------------------------------------------------
# Select strong negative ZFS seeds
# ----------------------------------------------------------

seed_df = df[df[zfs_col] <= -120].reset_index(drop=True)

print("Seed complexes:", len(seed_df))

# ----------------------------------------------------------
# Save seeds
# ----------------------------------------------------------

seed_df.to_csv("seed_complexes.csv", index=False)
