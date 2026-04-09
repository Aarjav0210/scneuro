#!/usr/bin/env python
"""
split_donors.py — Donor-level test holdout + K-fold cross-validation.

1. Carves off a stratified test set (--test fraction, default 15%).
2. Runs StratifiedKFold(K) on the remaining trainval donors.

Output CSV columns: donor_id, set (trainval/test), fold (0..K-1 or -1 for test), <stratify_col>

Usage:
    python split_donors.py <h5ad_path> [--kfold 3] 

Loading splits:
    df = pd.read_excel('donor_splits.xlsx', sheet_name='splits')
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
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit


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


def build_donor_df(cell_df, stratify_col):
    """Collapse cell-level rows to one row per donor, using the modal label."""
    n_unique = cell_df.groupby("donor_id")["strat_label"].nunique()
    if (n_unique > 1).any():
        print(f"WARNING: {(n_unique > 1).sum()} donor(s) have conflicting '{stratify_col}' labels — using mode.")

    donor_df = (
        cell_df.groupby("donor_id")["strat_label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    donor_df["n_cells"] = cell_df.groupby("donor_id").size().reindex(donor_df["donor_id"]).values

    missing = donor_df["strat_label"].isnull() | (donor_df["strat_label"].str.strip() == "")
    if missing.any():
        print(f"WARNING: {missing.sum()} donor(s) have missing '{stratify_col}' labels — excluded.")
        donor_df = donor_df[~missing].reset_index(drop=True)

    return donor_df


def print_dist(labels, header=None):
    """Print label counts and percentages."""
    if header:
        print(header)
    counts = pd.Series(labels).value_counts()
    for label, n in counts.items():
        print(f"    {label}: {n} ({n / len(labels) * 100:.1f}%)")


def collect_dist(labels, group, col):
    """Return rows for the stats sheet: one row per label value."""
    series = pd.Series(labels)
    counts = series.value_counts()
    return [
        {"group": group, "col": col, "label": label,
         "count": int(n), "pct": round(n / len(series) * 100, 1)}
        for label, n in counts.items()
    ]


def main():
    parser = argparse.ArgumentParser(description="Donor-level test holdout + K-fold CV from an h5ad file.")
    parser.add_argument("h5ad_path")
    parser.add_argument("--kfold",        type=int,   default=3,               metavar="K")
    parser.add_argument("--test",         type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--stratify-col", default="ADNC")
    parser.add_argument("--output",       default="donor_splits.xlsx")
    args = parser.parse_args()

    if args.kfold < 2:
        print("ERROR: --kfold must be >= 2")
        sys.exit(1)

    # ── 1. Load cell-level data ──────────────────────────────────────────────
    print(f"Reading: {args.h5ad_path}")
    with h5py.File(args.h5ad_path, "r") as f:
        donor_ids    = load_obs_column(f, "Donor ID")
        strat_labels = load_obs_column(f, args.stratify_col)
    print(f"  {len(donor_ids):,} cells")

    # ── 2. Aggregate to donor level ──────────────────────────────────────────
    donor_df = build_donor_df(
        pd.DataFrame({"donor_id": donor_ids, "strat_label": strat_labels}),
        args.stratify_col,
    )
    donors   = donor_df["donor_id"].values
    labels   = donor_df["strat_label"].values
    n_cells  = donor_df["n_cells"].values
    print(f"  {len(donors)} unique donors")
    print_dist(labels, header=f"\n{args.stratify_col} distribution (all donors):")

    label_counts = pd.Series(labels).value_counts()

    # ── 3. Carve off test set ────────────────────────────────────────────────
    if label_counts.min() >= 3:
        holdout = StratifiedShuffleSplit(n_splits=1, test_size=args.test, random_state=args.seed)
        tv_idx, test_idx = next(holdout.split(donors, labels))
    else:
        print(f"WARNING: small classes {label_counts[label_counts < 3].to_dict()} — unstratified test split.")
        holdout = ShuffleSplit(n_splits=1, test_size=args.test, random_state=args.seed)
        tv_idx, test_idx = next(holdout.split(donors))

    tv_donors, tv_labels = donors[tv_idx], labels[tv_idx]
    test_donors, test_labels = donors[test_idx], labels[test_idx]

    print(f"\nTest set : {len(test_donors)} donors ({len(test_donors)/len(donors)*100:.1f}%)")
    print(f"Train/val: {len(tv_donors)} donors ({len(tv_donors)/len(donors)*100:.1f}%)")

    # ── 4. K-fold CV on trainval ─────────────────────────────────────────────
    tv_label_counts = pd.Series(tv_labels).value_counts()
    if tv_label_counts.min() >= args.kfold:
        kf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed + 1)
        fold_iter = kf.split(tv_donors, tv_labels)
    else:
        print(f"WARNING: small classes in trainval {tv_label_counts[tv_label_counts < args.kfold].to_dict()} — unstratified KFold.")
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed + 1)
        fold_iter = kf.split(tv_donors)

    fold_assignment = np.empty(len(tv_donors), dtype=int)
    for fold, (_, val_idx) in enumerate(fold_iter):
        fold_assignment[val_idx] = fold

    # ── 5. Build and save output ─────────────────────────────────────────────
    tv_cells, test_cells = n_cells[tv_idx], n_cells[test_idx]
    out_df = pd.concat([
        pd.DataFrame({"donor_id": tv_donors,   "set": "trainval", "fold": fold_assignment, args.stratify_col: tv_labels,   "n_cells": tv_cells}),
        pd.DataFrame({"donor_id": test_donors, "set": "test",     "fold": -1,              args.stratify_col: test_labels, "n_cells": test_cells}),
    ], ignore_index=True)

    assert out_df["donor_id"].nunique() == len(out_df), "BUG: duplicate donor IDs in output"

    # ── 6. Summary + collect stats ───────────────────────────────────────────
    col = args.stratify_col
    stats_rows = collect_dist(labels, "all_donors", col)

    print(f"\n{col} distribution — test set:")
    print_dist(test_labels)
    stats_rows.extend(collect_dist(test_labels, "test", col))

    print(f"\nCV fold summary (k={args.kfold}, val donors per fold):")
    for fold in range(args.kfold):
        fold_labels = out_df[out_df["fold"] == fold][col]
        print_dist(fold_labels, header=f"  fold {fold} ({len(fold_labels)} donors):")
        stats_rows.extend(collect_dist(fold_labels, f"fold_{fold}", col))

    stats_df = pd.DataFrame(stats_rows, columns=["group", "col", "label", "count", "pct"])

    # ── 7. Save to Excel (two sheets) ────────────────────────────────────────
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="splits", index=False)
        stats_df.to_excel(writer, sheet_name="stats", index=False)

    print(f"\nSaved → {args.output}  (sheets: splits, stats)")


if __name__ == "__main__":
    main()
