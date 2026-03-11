import anndata as ad
import scanpy as sc
from scipy import sparse
import numpy as np
import os
import glob


INPUT = "/users/aiyer51/scratch/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
OUTDIR = "/users/aiyer51/scratch/qc_norm_mtg_by_donor"
COMBINED_OUT = os.path.join(OUTDIR, "combined.h5ad")
os.makedirs(OUTDIR, exist_ok=True)

MAX_CELLS_PER_DONOR = None
SCVI_KEY = "X_scVI"   # change if your embedding key is different


adata = sc.read_h5ad(INPUT)
print("Original:", adata)


cols_to_keep = [
    "library_prep",
    "Donor ID",
    "Method",
    "Sex",
    "Age at Death",
    "Race (choice=White)",
    "Race (choice=Black/ African American)",
    "Race (choice=Asian)",
    "Race (choice=American Indian/ Alaska Native)",
    "Race (choice=Native Hawaiian or Pacific Islander)",
    "Race (choice=Unknown or unreported)",
    "Race (choice=Other)",
    "Hispanic/Latino",
    "Years of education",
    "PMI",
    "APOE Genotype",
    "Thal",
    "Braak",
    "CERAD",
    "ADNC",
    "LATE",
    "Highest Lewy Body Disease",
    "Cognitive Status",
    "Class",
    "Subclass",
    "Supertype",
    "percent 6e10 positive area",
    "percent AT8 positive area",
    "percent NeuN positive area",
    "percent GFAP positive area",
    "percent aSyn positive area",
    "percent pTDP43 positive area",
]
adata.obs = adata.obs[cols_to_keep]

if SCVI_KEY not in adata.obsm:
    raise KeyError(
        f"{SCVI_KEY} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
    )

# preserve only scVI in obsm
scvi_mat = adata.obsm[SCVI_KEY].copy()
adata.obsm = {SCVI_KEY: scvi_mat}

adata.obsp = {}
adata.uns = {}

print("After trimming annotations:", adata)
print("Retained obsm keys:", list(adata.obsm.keys()))


donors = adata.obs["Donor ID"].unique().tolist()
print(f"Found {len(donors)} donors")

written_files = []

for donor in donors:
    print(f"\nprocessing donor: {donor}")

    donor_mask = (adata.obs["Donor ID"] == donor).values
    ad_d = adata[donor_mask, :].copy()
    print("donor subset:", ad_d)

    # Build donor object from raw UMI counts
    adata_raw = ad.AnnData(
        X=ad_d.layers["UMIs"].copy(),
        obs=ad_d.obs.copy(),
        var=ad_d.var.copy(),
        obsm={SCVI_KEY: ad_d.obsm[SCVI_KEY].copy()},
    )
    del ad_d

    if not sparse.issparse(adata_raw.X):
        adata_raw.X = sparse.csr_matrix(adata_raw.X)

    print(" Raw (sparse X):", adata_raw)
    print(" obsm keys:", list(adata_raw.obsm.keys()))

    adata_raw.var["mt"] = adata_raw.var_names.str.startswith("MT-")
    adata_raw.var["ribo"] = adata_raw.var_names.str.startswith(("RPS", "RPL"))

    sc.pp.calculate_qc_metrics(
        adata_raw,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    sc.pp.filter_cells(adata_raw, min_genes=200)
    sc.pp.filter_cells(adata_raw, min_counts=500)
    adata_raw = adata_raw[adata_raw.obs["pct_counts_mt"] < 5].copy()
    sc.pp.filter_genes(adata_raw, min_cells=10)
    print(" Post filters:", adata_raw)

    if MAX_CELLS_PER_DONOR is not None and adata_raw.n_obs > MAX_CELLS_PER_DONOR:
        idx = np.random.choice(adata_raw.n_obs, MAX_CELLS_PER_DONOR, replace=False)
        adata_raw = adata_raw[idx, :].copy()
        print(" After downsampling:", adata_raw)

    adata_raw.layers["counts"] = adata_raw.X.copy()

    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)

    print(" Final donor object:", adata_raw)

    safe_donor = str(donor).replace(" ", "_").replace("/", "_")
    out_path = os.path.join(
        OUTDIR, f"qc_norm_mtg.sparse_trimmed.donor_{safe_donor}.h5ad"
    )
    adata_raw.write_h5ad(out_path)
    written_files.append(out_path)
    print(f" Saved {out_path}")

    del adata_raw

print("All donors processed.")

#combine all donors
print("\nLoading donor files for merge...")
adatas = [sc.read_h5ad(f) for f in sorted(written_files)]

combined = ad.concat(
    adatas,
    axis=0,
    join="inner",
    merge="same",
    uns_merge="same",
    label=None,
    index_unique=None,
)

# Remove stale per-donor var stats if present
stale_var_cols = [
    "n_cells_by_counts",
    "mean_counts",
    "pct_dropout_by_counts",
    "total_counts",
    "n_cells",
]
for col in stale_var_cols:
    if col in combined.var.columns:
        del combined.var[col]

print("Combined object:", combined)
print("Combined obsm keys:", list(combined.obsm.keys()))

combined.write_h5ad(COMBINED_OUT)
print(f"Saved combined object to {COMBINED_OUT}")
