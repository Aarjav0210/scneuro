import scanpy as sc
import pandas as pd
import os
import gc


DONOR_DIR = os.getenv("SCRATCH_DATA_PATH", "./qc_norm_mtg_by_donor")

INPUT = os.path.join(DONOR_DIR, "combined.h5ad")
OUTDIR = os.path.join(DONOR_DIR, "hvg_outputs")
os.makedirs(OUTDIR, exist_ok=True)
COUNTS_LAYER = "counts"
HVG_LIST = [2000, 3000, 5000]
BATCH_KEY = "Donor ID"

adata = sc.read_h5ad(INPUT)
print("Loaded:", adata)


# Columns added by HVG; we'll strip these between runs
HVG_COLS = [
    "highly_variable", "means", "dispersions", "dispersions_norm",
    "highly_variable_rank", "highly_variable_nbatches", "highly_variable_intersection",
    "mean", "std",  # seurat flavor may add these
]

for n_hvg in HVG_LIST:
    print(f"\ntop {n_hvg}")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat_v3",
        layer=COUNTS_LAYER,
        subset=False,
        inplace=True,
        batch_key=BATCH_KEY,
    )

    # Save full annotated object
    full_out = os.path.join(OUTDIR, f"combined_hvg_annotated_top{n_hvg}.h5ad")
    adata.write_h5ad(full_out)
    print(f"Saved annotated object: {full_out}")

    ad_hvg = adata[:, adata.var["highly_variable"]].copy()
    subset_out = os.path.join(OUTDIR, f"combined_hvg_top{n_hvg}.h5ad")
    ad_hvg.write_h5ad(subset_out)
    print(f"Saved HVG-only object: {subset_out}")

    cols = ["highly_variable"]
    extra_cols = [
        "means", "variances", "variances_norm",
        "highly_variable_rank", "highly_variable_nbatches", "highly_variable_intersection",
    ]
    cols += [c for c in extra_cols if c in adata.var.columns]
    hvg_df = adata.var.loc[adata.var["highly_variable"], cols].copy()
    hvg_df.insert(0, "gene", hvg_df.index)
    csv_out = os.path.join(OUTDIR, f"hvg_top{n_hvg}.csv")
    hvg_df.to_csv(csv_out, index=False)
    print(f"HVG CSV: {csv_out}")

    del ad_hvg
    for col in HVG_COLS:
        if col in adata.var.columns:
            del adata.var[col]
    gc.collect()

print("\nDone.")
