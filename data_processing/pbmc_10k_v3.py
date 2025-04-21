# Import necessary libraries
import scanpy as sc
import pandas as pd
import numpy as np
import os # Used for creating output directory

# --- 1. Load Data ---

# Define file paths (adjust these to your actual file locations)
h5_path = "raw_data/10x_scrna/pbmc10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"
doublet_path = "raw_data/10x_scrna/pbmc10k_v3/matrix_doublets.tsv"
output_dir = "scanpy_objects"
output_file = os.path.join(output_dir, "pbmc_10k_v3.h5ad") # Scanpy typically uses .h5ad

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load 10x v3 pbmc data using scanpy.read_10x_h5
# This creates an AnnData object, the standard data structure in Scanpy.
adata = sc.read_10x_h5(filename=h5_path)

# Make gene names unique (equivalent to make.unique in R)
adata.var_names_make_unique()

# Load scrublet doublet scores
# Note: Seurat's AddMetaData adds columns to the metadata frame.
# In Scanpy, this metadata is stored in adata.obs.
scrublet_scores = pd.read_csv(doublet_path, sep="\t", names=['observed_doublet_score', 'simulated_doublet_score'], index_col=0)

# Ensure the scrublet index matches the AnnData object's cell barcodes (obs_names)
# This assumes the TSV file has barcodes as the first column (index_col=0)
# and they match the barcodes in the H5 file.
scrublet_scores = scrublet_scores.reindex(adata.obs_names)

# Add scrublet scores to adata.obs
adata.obs['observed_doublet_score'] = scrublet_scores['observed_doublet_score']
adata.obs['simulated_doublet_score'] = scrublet_scores['simulated_doublet_score'] # Optional, keep if needed

# Rename cells (equivalent to RenameCells with add.cell.id)
# This adds a prefix to the cell barcodes (adata.obs_names)
adata.obs_names = 'rna_' + adata.obs_names

# --- 2. Calculate Mitochondrial Percentage ---

# Identify mitochondrial genes (genes starting with "MT-")
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# Calculate QC metrics using scanpy.pp.calculate_qc_metrics
# This calculates n_genes_by_counts, total_counts, and pct_counts_mt
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Rename columns for clarity (optional, matches Seurat's naming)
adata.obs.rename(columns={'total_counts': 'nCount_RNA',
                          'n_genes_by_counts': 'nFeature_RNA',
                          'pct_counts_mt': 'percent.mito'}, inplace=True)
# Convert percent.mito to fraction (like in the R code)
adata.obs['percent.mito'] = adata.obs['percent.mito'] / 100

# --- 3. Quality Control (QC) ---

# Subset based on QC criteria (equivalent to Seurat's subset)
# Filter on counts, mitochondrial percentage, and doublet score
print(f"Cells before QC: {adata.n_obs}")
adata = adata[adata.obs['nCount_RNA'] > 2000, :]
adata = adata[adata.obs['nCount_RNA'] < 20000, :]
adata = adata[adata.obs['percent.mito'] < 0.2, :]
# Ensure the doublet score column name matches what was added earlier
adata = adata[adata.obs['observed_doublet_score'] < 0.1, :]
print(f"Cells after QC: {adata.n_obs}")

# Filter genes (equivalent to min.cells = 5 in CreateSeuratObject)
# Note: min.features = 500 was already implicitly handled by reading the
# filtered_feature_bc_matrix, but we apply min.cells here.
sc.pp.filter_genes(adata, min_cells=5)
print(f"Genes after filtering: {adata.n_vars}")


# --- 4. Preprocessing ---

# Normalize data (equivalent to NormalizeData with LogNormalize method)
# Normalizes counts per cell to 10,000, then log-transforms
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Find variable features (equivalent to FindVariableFeatures)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')

# Subset AnnData to highly variable genes for downstream steps like PCA
adata.raw = adata # Store the full (normalized, log-transformed) data in adata.raw
adata = adata[:, adata.var.highly_variable]

# Scale data (equivalent to ScaleData)
# Scales to unit variance and zero mean. Clips values to max_value 10.
sc.pp.scale(adata, max_value=10)

# Run PCA (equivalent to RunPCA)
sc.tl.pca(adata, n_comps=100, svd_solver='arpack') # arpack is often used for large datasets

# Run t-SNE (equivalent to RunTSNE)
# Uses the PCA results (default uses 'X_pca')
# Note: Seurat's dims=1:30 is handled here via n_pcs=30
sc.tl.tsne(adata, n_pcs=30)

# Find Neighbors (equivalent to FindNeighbors)
# Computes a neighborhood graph based on PCA space
# Note: Seurat's dims=1:30 is handled here via n_pcs=30
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30) # n_neighbors=10 is a common default, adjust if needed

# Find Clusters (equivalent to FindClusters)
# Uses the Louvain algorithm (default in Scanpy, algorithm=3 in Seurat is Leiden, but Louvain is closer)
# Note: Seurat's algorithm=3 corresponds to the Leiden algorithm.
# If Leiden is crucial, install `leidenalg` and use `sc.tl.leiden`.
# Here we use Louvain as it's built-in and commonly used.
sc.tl.louvain(adata, resolution=0.4, key_added='louvain_r0.4') # Store cluster info in adata.obs['louvain_r0.4']

# Run UMAP (equivalent to RunUMAP)
# Uses the neighborhood graph computed by sc.pp.neighbors
# Seurat's RunUMAP(graph='RNA_nn') uses the nearest neighbor graph.
# Scanpy's default uses the graph computed by sc.pp.neighbors, which is similar.
# Seurat's metric='euclidean' is implicitly used when FindNeighbors is run on PCA.
sc.tl.umap(adata)

# --- 5. Annotation ---

# Define new cluster IDs
# Ensure the order matches the cluster levels from Louvain (0, 1, 2, ...)
# Check adata.obs['louvain_r0.4'].cat.categories to confirm the order if needed
new_cluster_ids_map = {
    '0': "CD14+ Monocytes",
    '1': 'CD4 Memory',
    '2': 'CD4 Naive',
    '3': 'pre-B cell',
    '4': 'Double negative T cell',
    '5': 'NK cell',
    '6': 'B cell progenitor',
    '7': 'CD8 effector',
    '8': 'CD8 Naive',
    '9': 'CD16+ Monocytes',
    '10': 'Dendritic cell',
    '11': 'pDC',
    '12': 'Platelet'
    # Add more mappings if your clustering resulted in more clusters
}

# Add celltype annotation based on Louvain clusters
# Ensure the cluster column name ('louvain_r0.4') matches what was used in sc.tl.louvain
adata.obs['celltype'] = adata.obs['louvain_r0.4'].map(new_cluster_ids_map).astype('category')

# Reorder categories for plotting/consistency (optional)
ordered_celltypes = [
    "CD14+ Monocytes", 'CD4 Memory', 'CD4 Naive', 'pre-B cell',
    'Double negative T cell', 'NK cell', 'B cell progenitor',
    'CD8 effector', 'CD8 Naive', 'CD16+ Monocytes', 'Dendritic cell',
    'pDC', 'Platelet'
]
# Check if all expected cell types are present after mapping
present_celltypes = [ct for ct in ordered_celltypes if ct in adata.obs['celltype'].cat.categories]
adata.obs['celltype'] = adata.obs['celltype'].cat.reorder_categories(present_celltypes, ordered=True)


# Sub-annotate NK cells based on GZMK expression
# Get expression data for GZMK.
# We use adata.raw because adata currently only contains highly variable genes.
# adata.raw contains the normalized, log-transformed data for all genes.
nk_cells_mask = adata.obs['celltype'] == 'NK cell'
nk_cells_indices = adata.obs.index[nk_cells_mask]

# Ensure 'GZMK' is present in the raw variable names
if 'GZMK' in adata.raw.var_names:
    # Get GZMK expression for all cells from the raw data
    gzmk_expression = adata.raw[nk_cells_indices, 'GZMK'].X.toarray().flatten() # Use .toarray() if X is sparse

    # Create the 'bright' annotation based on expression threshold
    nk_bright_status = np.where(gzmk_expression > 1, 'NK bright', 'NK dim')

    # Update the 'celltype' column for NK cells
    # Use .loc for safe assignment based on index
    adata.obs.loc[nk_cells_indices, 'celltype'] = nk_bright_status

    # Add the new categories to the 'celltype' categorical type
    adata.obs['celltype'] = adata.obs['celltype'].astype(str) # Convert to string to add new categories easily
    new_categories = list(adata.obs['celltype'].unique())
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    adata.obs['celltype'] = adata.obs['celltype'].cat.reorder_categories(new_categories, ordered=False) # Re-categorize

    print("NK cells sub-annotated based on GZMK expression.")
else:
    print("Warning: Gene 'GZMK' not found in adata.raw.var_names. Skipping NK cell sub-annotation.")


# --- 6. Save Object ---

# Save the processed AnnData object
# The .h5ad format stores the matrix, annotations, and analysis results
adata.write(output_file)

print(f"\nAnalysis complete. Processed data saved to: {output_file}")
print("\nFinal AnnData object overview:")
print(adata)
# You can also visualize results, e.g.:
# sc.pl.umap(adata, color=['louvain_r0.4', 'celltype'], save='_clusters_celltypes.png')