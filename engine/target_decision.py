import pandas as pd

def decide_from_database(
    df: pd.DataFrame,
    zfs_col: str,
    target_zfs: float,
    tol: float = 1.0
):
    """
    Check if target ZFS exists in database within tolerance.
    """

    df = df.copy()
    df["delta"] = (df[zfs_col] - target_zfs).abs()

    hits = df[df["delta"] <= tol].sort_values("delta")

    if len(hits) > 0:
        return True, hits

    return False, None
