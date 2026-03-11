import anndata as ad
import scanpy as sc
import os
import glob


DONOR_DIR = "/users/aiyer51/scratch/qc_norm_mtg_by_donor"
OUT_COMBINED = os.path.join(DONOR_DIR, "combined.h5ad")


donor_files = sorted(glob.glob(os.path.join(DONOR_DIR, "qc_norm_mtg.sparse_trimmed.donor_*.h5ad")))
print(f"Found {len(donor_files)} donor files to merge")

adatas = []
for fpath in donor_files:
    donor_id = os.path.basename(fpath).replace("qc_norm_mtg.sparse_trimmed.donor_", "").replace(".h5ad", "")
    print(f"{donor_id} processing")
    adata_d = sc.read_h5ad(fpath)
    adatas.append(adata_d)


print("\nConcatenating ...")
combined = ad.concat(
    adatas,
    axis=0,
    join="outer",          # keeps all genes
    merge="same",          # keeps shared var columns (gene_ids, mt, ribo, etc.)
    uns_merge="same",
    label=None,            # no extra batch column (Donor ID already in obs)
)

# Clean up any per-donor var stats that are now stale after concatenation
stale_var_cols = ["n_cells_by_counts", "mean_counts", "pct_dropout_by_counts",
                  "total_counts", "n_cells"]
for col in stale_var_cols:
    if col in combined.var.columns:
        del combined.var[col]

print("Combined object:", combined)
print("obs columns :", combined.obs.columns.tolist())
print("var columns :", combined.var.columns.tolist())
print("layers      :", list(combined.layers.keys()))


print(f"\nWriting to {OUT_COMBINED} ...")
combined.write_h5ad(OUT_COMBINED)
print("done!!!!!!!!!!")
