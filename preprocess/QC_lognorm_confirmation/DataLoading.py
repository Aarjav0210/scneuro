import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as so

adata = sc.read_h5ad("/users/aiyer51/scratch/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad")

print(adata)
print("\nobs columns:\n", list(adata.obs.columns))
print("var columns:\n",  list(adata.var.columns))
print("uns keys:\n",     list(adata.uns.keys()))
print("layers:\n",       list(adata.layers.keys()))

#raw counts or normalized

#per cell qc metrics
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt", "ribo"],
    percent_top=None,
    log1p=False,
    inplace=True
)

print("\nQC metric summary:")
print(adata.obs[["n_genes_by_counts",
                  "total_counts",
                  "pct_counts_mt",
                  "pct_counts_ribo"]].describe().round(2))

important_cols = [
    "cell_type", "subclass_label", "supertype_label", "class_label",
    "donor_id", "sample_id", "Diagnosis", "ADNC",
    "Braak_stage", "CERAD", "CPS",
    "doublet_score", "low_quality", "outlier",
    "mapping_confidence", "cluster_confidence_score"
]
present = [c for c in important_cols if c in adata.obs.columns]
print("\nPresent key columns:", present)
if present:
    print(adata.obs[present].describe(include="all").T[["count","unique","top","freq"]].to_string())

for col in ["ADNC", "CERAD", "Braak", "Cognitive Status", "LATE"]:
    print(f"\n{col} ")
    print(adata.obs[col].value_counts())

donor_counts = adata.obs.groupby("Donor ID").size().sort_values(ascending=False)
print(donor_counts.describe())
print(f"Top 5 donors:\n{donor_counts.head()}")

print(adata.obs.groupby(["Donor ID","ADNC"]).size().unstack(fill_value=0))

ct_props = (adata.obs.groupby(["Donor ID","Class"])
              .size().unstack(fill_value=0)
              .apply(lambda r: r/r.sum(), axis=1))
print(ct_props.round(3))

print("batch_condition:", adata.uns.get("batch_condition"))
print(adata.obs["Method"].value_counts())
print(adata.obs["library_prep"].value_counts())

#day 3
import pandas as pd
# Are any donors exclusively Multiome?
print(adata.obs.groupby(["Donor ID","Method"]).size().unstack(fill_value=0))

#donor cell count imbalance
donor_counts = adata.obs.groupby("Donor ID", observed=True).size()
low_cell_donors = donor_counts[donor_counts < 5000]
print(f"Donors with <5k cells: {len(low_cell_donors)}")
print(low_cell_donors.sort_values())
