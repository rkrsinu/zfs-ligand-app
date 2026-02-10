import pandas as pd

# ----------------------------------------------------------
# Load opt_D.csv (NOT GA.csv)
# ----------------------------------------------------------
df = pd.read_csv("opt_D.csv")

# ----------------------------------------------------------
# Select strong negative ZFS seeds
# (same logic as before, correct column name)
# ----------------------------------------------------------
seed_df = df[df["opt_zfs"] <= -120].reset_index(drop=True)

print("Seed complexes:", len(seed_df))

# ----------------------------------------------------------
# Save seeds
# ----------------------------------------------------------
seed_df.to_csv("seed_complexes.csv", index=False)
