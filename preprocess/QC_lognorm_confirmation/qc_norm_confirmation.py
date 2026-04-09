import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.stats as stats

adata = sc.read_h5ad("/users/aiyer51/scratch/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad")

adata.var["mt"]   = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt", "ribo"],
    percent_top=None,
    log1p=False,
    inplace=True
)

np.random.seed(42)
idx = np.random.choice(adata.n_obs, size=5000, replace=False)
sub = adata[idx]

X_sub = sub.X
if sp.issparse(X_sub): X_sub = X_sub.toarray()

UMI_sub = sub.layers["UMIs"]
if sp.issparse(UMI_sub): UMI_sub = UMI_sub.toarray()

cell_totals_X   = X_sub.sum(axis=1)
cell_totals_UMI = UMI_sub.sum(axis=1)

gene_max  = UMI_sub.max(axis=0).flatten()
gene_idx  = int(np.where(gene_max > 10)[0][0])
gene_name = adata.var_names[gene_idx]
x_vals    = X_sub[:, gene_idx].flatten()
umi_vals  = UMI_sub[:, gene_idx].flatten()

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("SEAAD QC Verification", fontsize=16, fontweight="bold")

# Point 1a: n_genes histogram
ax = axes[0, 0]
ax.hist(adata.obs["n_genes_by_counts"], bins=100, color="steelblue", edgecolor="none")
ax.axvline(200, color="red", linestyle="--", label="Typical filter cutoff (200)")
ax.set_xlabel("n_genes_by_counts")
ax.set_ylabel("# Cells")
ax.set_title("Point 1a: Gene count distribution\n(no tail below 200 = already filtered)")
ax.legend()

# Point 1b: MT% histogram
ax = axes[0, 1]
ax.hist(adata.obs["pct_counts_mt"], bins=100, color="salmon", edgecolor="none")
ax.axvline(5,  color="red",     linestyle="--", label="snRNA-seq MT threshold (5%)")
ax.axvline(20, color="darkred", linestyle="--", label="scRNA-seq MT threshold (20%)")
ax.set_xlabel("% Mitochondrial counts")
ax.set_ylabel("# Cells")
ax.set_title("Point 1b: MT% distribution\n(no high-MT tail = already filtered)")
ax.legend()

# Point 1c: total_counts on X
ax = axes[0, 2]
ax.hist(adata.obs["total_counts"], bins=100, color="mediumpurple", edgecolor="none")
ax.set_xlabel("total_counts (sum of X per cell)")
ax.set_ylabel("# Cells")
ax.set_title("Point 1c: Total counts distribution on X\n(tight = normalized, not raw)")

# Point 2a: per-cell total in X
ax = axes[1, 0]
ax.hist(cell_totals_X, bins=80, color="mediumpurple", edgecolor="none", alpha=0.8, label="X (normalized)")
ax.set_xlabel("Sum of values per cell")
ax.set_ylabel("# Cells")
ax.set_title("Point 2a: Per-cell total in X\n(tight = normalized)")
ax.legend()

# Point 2b: per-cell total in UMIs
ax = axes[1, 1]
ax.hist(cell_totals_UMI, bins=80, color="darkorange", edgecolor="none", alpha=0.8, label="UMIs layer (raw)")
ax.set_xlabel("Sum of UMIs per cell")
ax.set_ylabel("# Cells")
ax.set_title("Point 2b: Per-cell total in UMIs layer\n(wide + right-skewed = raw counts)")
ax.legend()

# Point 2c: overlay
ax = axes[1, 2]
ax.hist(cell_totals_X,   bins=80, color="mediumpurple", alpha=0.6, label="X (normalized)", density=True)
ax.hist(cell_totals_UMI, bins=80, color="darkorange",   alpha=0.6, label="UMIs (raw)",     density=True)
ax.set_xlabel("Sum per cell")
ax.set_ylabel("Density")
ax.set_title("Point 2c: X vs UMIs overlay\n(shape difference confirms normalization)")
ax.legend()

# Point 3a: raw UMI values for one gene
ax = axes[2, 0]
ax.hist(umi_vals[umi_vals > 0], bins=40, color="darkorange", edgecolor="none")
ax.set_xlabel("UMI count")
ax.set_ylabel("# Cells")
ax.set_title(f"Point 3a: Raw UMIs for {gene_name}\n(integers, right-skewed)")

# Point 3b: normalized X values for same gene
ax = axes[2, 1]
ax.hist(x_vals[x_vals > 0], bins=40, color="mediumpurple", edgecolor="none")
ax.set_xlabel("Normalized value")
ax.set_ylabel("# Cells")
ax.set_title(f"Point 3b: Normalized X for {gene_name}\n(continuous, log-compressed)")

# Point 3c: Q-Q plot of X values
ax = axes[2, 2]
nonzero_x = x_vals[x_vals > 0]
stats.probplot(nonzero_x, dist="norm", plot=ax)
ax.set_title("Point 3c: Q-Q plot of X values\n(fits normal = log1p normalized)")

plt.tight_layout()
plt.savefig("qc_verification.png", dpi=150, bbox_inches="tight")
print("Saved: qc_verification.png")

print(f"X    per-cell sum:  mean={cell_totals_X.mean():.1f},  std={cell_totals_X.std():.1f},  CV={cell_totals_X.std()/cell_totals_X.mean():.3f}")
print(f"UMIs per-cell sum:  mean={cell_totals_UMI.mean():.1f}, std={cell_totals_UMI.std():.1f}, CV={cell_totals_UMI.std()/cell_totals_UMI.mean():.3f}")
print(f"\nX    non-integer fraction: {(~np.isclose(X_sub, X_sub.round())).mean():.4f}")
print(f"UMIs non-integer fraction: {(~np.isclose(UMI_sub, UMI_sub.round())).mean():.4f}")
print(f"\nuns['X_normalization']: {adata.uns.get('X_normalization', 'NOT FOUND')}")
