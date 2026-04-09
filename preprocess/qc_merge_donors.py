import anndata as ad
import scanpy as sc
import os
import glob
from scipy.sparse import csr_matrix  # Import required for the sparse safety check
from dotenv import load_dotenv

load_dotenv()


DONOR_DIR = os.environ["SCRATCH_DATA_PATH"]
OUT_COMBINED = os.environ["INPUT_COMBINED"]

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

#Safety check to ensure outer join didn't densify the matrix
print("Ensuring matrices are sparse...")
if not isinstance(combined.X, csr_matrix):
    combined.X = csr_matrix(combined.X)

for layer in combined.layers:
    if not isinstance(combined.layers[layer], csr_matrix):
        combined.layers[layer] = csr_matrix(combined.layers[layer])

print("Combined object:", combined)
print("obs columns :", combined.obs.columns.tolist())
print("var columns :", combined.var.columns.tolist())
print("layers      :", list(combined.layers.keys()))

print(f"\nWriting to {OUT_COMBINED} ...")
#GZIP compression to shrink the file size!
combined.write_h5ad(OUT_COMBINED, compression="gzip")
print("done!!!!!!!!!!")