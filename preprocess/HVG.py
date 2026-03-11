import scanpy as sc
import pandas as pd
import os


INPUT = "/users/aiyer51/scratch/qc_norm_mtg_by_donor/combined.h5ad"
OUTDIR = "/users/aiyer51/scratch/qc_norm_mtg_by_donor/hvg_outputs"
os.makedirs(OUTDIR, exist_ok=True)

HVG_LIST = [2000, 3000, 5000]


BATCH_KEY = "Donor ID"


adata = sc.read_h5ad(INPUT)
print("Loaded:", adata)
print("obs columns:", adata.obs.columns.tolist())
print("var columns:", adata.var.columns.tolist())


for n_hvg in HVG_LIST:
    print(f"\ntop {n_hvg}")

    ad = adata.copy()

    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=n_hvg,
        flavor="seurat",#seurat v3 need unnormalized data. should we use that instead?
        subset=False,
        inplace=True,
        batch_key=BATCH_KEY,
    )

    full_out = os.path.join(OUTDIR, f"combined_hvg_annotated_top{n_hvg}.h5ad")
    ad.write_h5ad(full_out)
    print(f"Saved annotated object: {full_out}")

    # subset object with HVGs only
    ad_hvg = ad[:, ad.var["highly_variable"]].copy()
    subset_out = os.path.join(OUTDIR, f"combined_hvg_top{n_hvg}.h5ad")
    ad_hvg.write_h5ad(subset_out)
    print(f"Saved HVG-only object: {subset_out}")

    # CSV of HVG genes and stats
    cols = ["highly_variable"]
    extra_cols = [
        "means",
        "dispersions",
        "dispersions_norm",
        "highly_variable_rank",
        "highly_variable_nbatches",
        "highly_variable_intersection",
    ]
    cols += [c for c in extra_cols if c in ad.var.columns]

    hvg_df = ad.var.loc[ad.var["highly_variable"], cols].copy()
    hvg_df.insert(0, "gene", hvg_df.index)

    csv_out = os.path.join(OUTDIR, f"hvg_top{n_hvg}.csv")
    hvg_df.to_csv(csv_out, index=False)
    print(f"HVG CSV: {csv_out}")

print("\nDone.")
