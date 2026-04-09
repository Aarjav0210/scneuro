#!/usr/bin/env python
"""
split_donors_multilabel.py — Donor-level test holdout + K-fold CV with joint stratification
on Braak, Thal, and CERAD using MultilabelStratifiedKFold.

Usage:
    python split_donors_multilabel.py <h5ad_path> [--kfold 5] 

Loading splits:
    df = pd.read_excel('donor_splits_multilabel.xlsx', sheet_name='splits')
    test_donors = df[df['set'] == 'test']['donor_id'].values
    for fold in range(K):
        train_donors = df[(df['set'] == 'trainval') & (df['fold'] != fold)]['donor_id'].values
        val_donors   = df[(df['set'] == 'trainval') & (df['fold'] == fold)]['donor_id'].values
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

STRATIFY_COLS = ["Braak", "Thal", "CERAD"]


def load_obs_column(f, key):
    """Load a single obs column from an open h5py File handle.

    Handles both categorical (codes + categories) and plain string/numeric arrays.
    """
    if key not in f["obs"]:
        raise KeyError(f"Column '{key}' not found in obs. Available: {list(f['obs'].keys())}")
    obj = f["obs"][key]
    if "codes" in obj and "categories" in obj:
        return obj["categories"][:].astype("U")[obj["codes"][:]]
    data = obj[:]
    return data.astype("U") if data.dtype.kind in ("S", "O") else data


def build_donor_df(cell_df, cols):
    """Collapse cell-level rows to one row per donor.

    For each stratify col, take the modal value across all cells for that donor.
    Drops donors missing any label. Adds n_cells column.
    """
    donor_df = pd.DataFrame({"donor_id": cell_df["donor_id"].unique()})
    donor_df = donor_df.set_index("donor_id")

    for col in cols:
        n_unique = cell_df.groupby("donor_id")[col].nunique()
        if (n_unique > 1).any():
            print(f"WARNING: {(n_unique > 1).sum()} donor(s) have conflicting '{col}' labels — using mode.")
        donor_df[col] = cell_df.groupby("donor_id")[col].agg(lambda x: x.mode()[0])

    donor_df["n_cells"] = cell_df.groupby("donor_id").size()
    donor_df = donor_df.reset_index()

    missing = donor_df[cols].isnull().any(axis=1) | (donor_df[cols] == "").any(axis=1)
    if missing.any():
        print(f"WARNING: {missing.sum()} donor(s) have missing labels — excluded.")
        donor_df = donor_df[~missing].reset_index(drop=True)

    return donor_df


def make_label_matrix(donor_df, cols):
    """One-hot encode each stratify col and concatenate into a 2D label matrix Y.

    Shape: (n_donors, sum of unique values across all cols)
    """
    return np.hstack([pd.get_dummies(donor_df[col]).values for col in cols])


def print_dist(series, header=None):
    """Print label counts and percentages for a single column."""
    if header:
        print(header)
    counts = series.value_counts()
    for label, n in counts.items():
        print(f"    {label}: {n} ({n / len(series) * 100:.1f}%)")


def collect_dist(series, group, col):
    """Return rows for the stats sheet: one row per label value."""
    counts = series.value_counts()
    return [
        {"group": group, "col": col, "label": label,
         "count": int(n), "pct": round(n / len(series) * 100, 1)}
        for label, n in counts.items()
    ]


def main():
    parser = argparse.ArgumentParser(description="Donor-level multilabel stratified split (Braak + Thal + CERAD).")
    parser.add_argument("h5ad_path")
    parser.add_argument("--kfold",  type=int,   default=3,    metavar="K")
    parser.add_argument("--test",   type=float, default=0.15)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--output", default="donor_splits_multilabel.xlsx")
    args = parser.parse_args()

    if args.kfold < 2:
        print("ERROR: --kfold must be >= 2")
        sys.exit(1)

    # ── 1. Load cell-level data ──────────────────────────────────────────────
    print(f"Reading: {args.h5ad_path}")
    with h5py.File(args.h5ad_path, "r") as f:
        cell_data = {"donor_id": load_obs_column(f, "Donor ID")}
        for col in STRATIFY_COLS:
            cell_data[col] = load_obs_column(f, col)
    cell_df = pd.DataFrame(cell_data)
    print(f"  {len(cell_df):,} cells")

    # ── 2. Aggregate to donor level ──────────────────────────────────────────
    donor_df = build_donor_df(cell_df, STRATIFY_COLS)
    print(f"  {len(donor_df)} unique donors")

    Y = make_label_matrix(donor_df, STRATIFY_COLS)
    donors = donor_df["donor_id"].values

    # ── 3. Carve off test set ────────────────────────────────────────────────
    # iterstrat has no ShuffleSplit equivalent, so we use KFold with
    # n_splits = round(1/test_size) and treat fold 0 as the test set.
    n_splits_holdout = round(1 / args.test)
    holdout = MultilabelStratifiedKFold(n_splits=n_splits_holdout, shuffle=True, random_state=args.seed)
    tv_idx, test_idx = next(holdout.split(donors, Y))

    tv_df   = donor_df.iloc[tv_idx].reset_index(drop=True)
    test_df = donor_df.iloc[test_idx].reset_index(drop=True)
    Y_tv    = Y[tv_idx]

    print(f"\nTest set : {len(test_df)} donors ({len(test_df)/len(donor_df)*100:.1f}%)")
    print(f"Train/val: {len(tv_df)} donors ({len(tv_df)/len(donor_df)*100:.1f}%)")

    # ── 4. K-fold CV on trainval ─────────────────────────────────────────────
    kf = MultilabelStratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed + 1)
    fold_assignment = np.empty(len(tv_df), dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(tv_df["donor_id"].values, Y_tv)):
        fold_assignment[val_idx] = fold

    # ── 5. Build and save output ─────────────────────────────────────────────
    tv_df["set"]  = "trainval"
    tv_df["fold"] = fold_assignment
    test_df["set"]  = "test"
    test_df["fold"] = -1

    out_df = pd.concat([tv_df, test_df], ignore_index=True)
    out_df = out_df[["donor_id", "set", "fold"] + STRATIFY_COLS + ["n_cells"]]

    assert out_df["donor_id"].nunique() == len(out_df), "BUG: duplicate donor IDs in output"

    # ── 6. Summary + collect stats ───────────────────────────────────────────
    stats_rows = []

    for col in STRATIFY_COLS:
        print_dist(donor_df[col], header=f"\n{col} distribution (all donors):")
        stats_rows.extend(collect_dist(donor_df[col], "all_donors", col))

    print(f"\nDistributions — test set ({len(test_df)} donors):")
    for col in STRATIFY_COLS:
        print_dist(test_df[col], header=f"  {col}:")
        stats_rows.extend(collect_dist(test_df[col], "test", col))

    print(f"\nCV fold summary (k={args.kfold}):")
    for fold in range(args.kfold):
        fold_df = out_df[out_df["fold"] == fold]
        print(f"  fold {fold} ({len(fold_df)} donors):")
        for col in STRATIFY_COLS:
            print_dist(fold_df[col], header=f"    {col}:")
            stats_rows.extend(collect_dist(fold_df[col], f"fold_{fold}", col))

    stats_df = pd.DataFrame(stats_rows, columns=["group", "col", "label", "count", "pct"])

    # ── 7. Save to Excel (two sheets) ────────────────────────────────────────
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="splits", index=False)
        stats_df.to_excel(writer, sheet_name="stats", index=False)

    print(f"\nSaved → {args.output}  (sheets: splits, stats)")


if __name__ == "__main__":
    main()
