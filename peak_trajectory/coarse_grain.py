from typing import Optional, Literal # Added Literal
import numpy as np
import pandas as pd # Added pandas
import scanpy as sc
from sklearn.cluster import KMeans
from peak_trajectory.util.input_validation import validate_matrix
import warnings # Added warnings

def select_top_peaks(
        adata: sc.AnnData,
        layer: Optional[str] = None, # Layer for variability calculation (e.g., lognorm X or counts)
        counts_layer: Optional[str] = 'counts', # Layer for accessibility filtering (raw counts preferred)
        min_accessibility_percent: Optional[float] = 0.01, # Min fraction of cells where peak is detected
        max_accessibility_percent: Optional[float] = 0.5, # Max fraction of cells where peak is detected
        hvg_method: Literal['scanpy', 'cell_ranger'] = 'scanpy', # Method for HVG selection
        n_top_peaks: Optional[int] = 5000, # Number of variable peaks to select by dispersion/variance
        # Parameters for scanpy's highly_variable_genes (if hvg_method='scanpy')
        hvg_min_mean: Optional[float] = 0.01, # Adjusted default lower than RNA
        hvg_max_mean: Optional[float] = 2.0,  # Adjusted default
        hvg_min_disp: Optional[float] = 0.25, # Adjusted default based on ATAC typical values
) -> np.ndarray:
    """
    Select variable peaks for peak-peak distance computation, inspired by common
    scATAC-seq processing workflows.

    Combines filtering by detection rate across cells (using counts) and
    selection of highly variable peaks (using specified layer, often log-normalized).

    :param adata: AnnData object (cells x peaks).
    :param layer: Layer for variability calculation (e.g., lognorm adata.X or 'counts').
                  If None, uses adata.X (expected to be log-normalized).
    :param counts_layer: Layer containing raw or suitably binarizable counts for
                         calculating accessibility percentages. If None, accessibility
                         filtering is skipped. Defaults to 'counts'.
    :param min_accessibility_percent: Minimum fraction of cells where peak must be detected
                                      (non-zero in counts_layer) to be kept. If None, no min filter.
    :param max_accessibility_percent: Maximum fraction of cells where peak can be detected.
                                      Helps remove ubiquitous peaks. If None, no max filter.
    :param hvg_method: Method for selecting Highly Variable Peaks.
                       'scanpy': uses sc.pp.highly_variable_genes.
                       'cell_ranger': mimics basic Cell Ranger filtering (less common now).
                       Set n_top_peaks=None to disable HVG selection.
    :param n_top_peaks: Target number of top variable peaks to select based on dispersion/variance.
                        If None, HVG selection step is skipped.
    :param hvg_min_mean: Min mean for scanpy HVG selection (on the `layer` data).
    :param hvg_max_mean: Max mean for scanpy HVG selection.
    :param hvg_min_disp: Min dispersion for scanpy HVG selection.
    :return: An array of selected peak names/identifiers.
    """
    if layer is not None and layer not in adata.layers.keys():
        raise ValueError(f"Layer '{layer}' for variability calculation not found. Available: {list(adata.layers.keys())}")
    if counts_layer is not None and counts_layer not in adata.layers.keys():
        raise ValueError(f"Layer '{counts_layer}' for accessibility filtering not found. Available: {list(adata.layers.keys())}")

    # --- 1. Accessibility Filtering (based on counts_layer) ---
    initial_peak_set = adata.var_names.to_numpy()
    if counts_layer is not None and (min_accessibility_percent is not None or max_accessibility_percent is not None):
        print(f"Filtering peaks by accessibility using layer '{counts_layer}'...")
        counts_matrix = adata.layers[counts_layer]

        # Calculate n_cells per peak (handling sparse)
        if hasattr(counts_matrix, 'getnnz'): # Sparse matrix
            # More efficient for sparse: sum of boolean representation
            binary_counts = counts_matrix > 0
            n_cells_accessible = np.array(binary_counts.sum(axis=0)).flatten()
        else: # Dense matrix
            n_cells_accessible = np.count_nonzero(counts_matrix, axis=0)

        percent_cells_accessible = n_cells_accessible / adata.n_obs
        adata.var['percent_cells_accessible'] = percent_cells_accessible # Store for info

        min_cells = 0 if min_accessibility_percent is None else int(min_accessibility_percent * adata.n_obs)
        max_cells = adata.n_obs if max_accessibility_percent is None else int(max_accessibility_percent * adata.n_obs)

        # Apply filter
        accessibility_mask = (n_cells_accessible >= min_cells) & (n_cells_accessible <= max_cells)
        peaks_after_accessibility_filter = adata.var_names[accessibility_mask].to_numpy()
        print(f"  Kept {len(peaks_after_accessibility_filter)} peaks after accessibility filter.")
        if len(peaks_after_accessibility_filter) == 0:
             raise ValueError("No peaks passed accessibility filtering. Adjust parameters.")
        current_peak_set = peaks_after_accessibility_filter
    else:
        print("Skipping accessibility filtering.")
        current_peak_set = initial_peak_set

    # --- 2. Highly Variable Peak Selection ---
    if n_top_peaks is None or hvg_method is None:
         print("Skipping Highly Variable Peak selection.")
         selected_peaks = current_peak_set
    else:
        print(f"Selecting top {n_top_peaks} variable peaks using method '{hvg_method}' and layer '{layer if layer else 'X'}'...")
        # Subset adata temporarily to only include peaks that passed accessibility filter
        adata_filt = adata[:, current_peak_set].copy()

        if hvg_method == 'scanpy':
             # Use scanpy's HVG selection
             # Note: This expects log-normalized data typically in X or the specified layer
             sc.pp.highly_variable_genes(
                 adata_filt,
                 layer=layer, # Use specified layer or default X
                 n_top_genes=n_top_peaks, # Use n_top_genes param for peaks
                 min_mean=hvg_min_mean,
                 max_mean=hvg_max_mean,
                 min_disp=hvg_min_disp,
                 flavor='seurat_v3' # Common flavor, alternatives exist
             )
             hvg_mask = adata_filt.var['highly_variable']
             selected_peaks = adata_filt.var_names[hvg_mask].to_numpy()

             # Check if enough HVGs were found
             if len(selected_peaks) < n_top_peaks:
                  warnings.warn(f"Found only {len(selected_peaks)} HVGs passing criteria, less than requested {n_top_peaks}.")
             if len(selected_peaks) == 0:
                  raise ValueError("No highly variable peaks found. Adjust HVG parameters or input layer.")

        elif hvg_method == 'cell_ranger':
             # Simplified version mimicking Cell Ranger filtering (less common for downstream)
             # Requires dispersion calculation if not present
             if 'dispersions_norm' not in adata_filt.var.columns:
                  # Calculate mean/dispersion on the specified layer
                   layer_data = adata_filt.X if layer is None else adata_filt.layers[layer]
                   # Ensure data is not log-transformed if starting from counts
                   # This part might need adjustment based on input layer data type
                   if np.expm1(layer_data.max()) > 100: # Heuristic check for log data
                         raise ValueError("Cell Ranger HVG expects non-log data, but input layer seems log-transformed.")
                   sc.pp.normalize_per_cell(adata_filt, counts_per_cell_after=1e4, layer=None) # Use X for calculation
                   sc.pp.log1p(adata_filt)
                   sc.pp.highly_variable_genes(adata_filt, layer=None, n_top_genes=n_top_peaks, flavor='cell_ranger') # Calculate dispersions

             hvg_mask = adata_filt.var['highly_variable']
             selected_peaks = adata_filt.var_names[hvg_mask].to_numpy()

        else:
             raise ValueError(f"Unknown hvg_method: {hvg_method}")

        print(f"  Selected {len(selected_peaks)} highly variable peaks.")
        # Combine filters implicitly as HVG was run on already filtered data
        # final_selected_peaks = np.intersect1d(peaks_after_accessibility_filter, hvg_peaks) # Not needed due to subsetting

    if len(selected_peaks) == 0:
         raise ValueError("No peaks selected after all filtering steps.")

    print(f"Total selected peaks: {len(selected_peaks)}")
    return selected_peaks


def coarse_grain(
        cell_embedding: np.ndarray,
        peak_accessibility: np.ndarray, # Renamed from gene_expression
        graph_dist: np.ndarray,
        n: int = 1000,
        cluster: Optional[np.array] = None,
        random_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply coarse-graining to reduce the number of cells [cite: 103]

    :param cell_embedding: the cell embedding [cite: 103]
    :param peak_accessibility: the peak accessibility matrix (cells x peaks) # Renamed [cite: 103]
    :param graph_dist: the graph distance matrix [cite: 103]
    :param n: number of coarse-grained cells (metacells) to keep [cite: 103]
    :param cluster: specify an array to use precomputed clusters. If not specified a KMeans clustering will be performed [cite: 103]
    :param random_seed: the random seed [cite: 103]
    :return: the updated peak accessibility and graph distance matrices for metacells # Modified [cite: 103]
    """
    # Renamed variable, ensure check is appropriate for accessibility data (non-negative)
    validate_matrix(peak_accessibility, obj_name='Peak Accessibility Matrix', min_value=0) # [cite: 103]
    ncells, nfeatures = peak_accessibility.shape # nfeatures is now npeaks

    validate_matrix(cell_embedding, obj_name='Cell embedding', nrows=ncells) # [cite: 104]

    if cluster is None:
        # Ensure cell_embedding is suitable for KMeans (e.g., PCA, LSI)
        k_means = KMeans(n_clusters=n, random_state=random_seed, n_init=10).fit(cell_embedding) # Added n_init explicitly
        cluster = k_means.labels_ # [cite: 104]

    knn_membership = np.zeros((n, cell_embedding.shape[0]))
    for i, c in enumerate(cluster):
        # Ensure cluster labels are within [0, n-1]
        if 0 <= c < n:
             knn_membership[c, i] = 1
        # else: handle potential outlier clusters if necessary

    # Sum accessibility for metacells
    peak_accessibility_updated = knn_membership @ peak_accessibility # Renamed [cite: 104]

    # Normalize membership for weighted average distance calculation
    cluster_sizes = np.sum(knn_membership, axis=1)
    # Avoid division by zero for empty clusters
    valid_clusters = cluster_sizes > 0
    knn_membership_norm = np.zeros_like(knn_membership)
    knn_membership_norm[valid_clusters, :] = knn_membership[valid_clusters, :] / cluster_sizes[valid_clusters, None] #[cite: 104]

    graph_dist_updated = knn_membership_norm @ graph_dist @ knn_membership_norm.T #[cite: 104]

    # Return only results for non-empty clusters if desired, or handle downstream
    # Here, we return the full n x npeaks and n x n matrices
    return peak_accessibility_updated, graph_dist_updated # Renamed


# Renamed from coarse_grain_adata
def coarse_grain_adata_peaks(
        adata: sc.AnnData, # [cite: 105]
        graph_dist: np.array,
        features: list[str], # Should be list of selected peak names
        n: int = 1000,
        reduction: str = "X_dm", # Or X_lsi, X_pca appropriate for ATAC [cite: 105]
        dims: int = 5, # [cite: 105]
        layer: Optional[str] = None, # Specify layer for peak accessibility
        random_seed=1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply coarse-graining using data from an AnnData object (adapted for peaks).

    :param adata: a scanpy Anndata object (cells x peaks) # Modified [cite: 105]
    :param graph_dist: the cell-cell graph distance matrix # [cite: 106]
    :param features: the features (i.e., peaks) to keep # Renamed [cite: 106]
    :param n: number of coarse-grained cells (metacells) to keep (default = 1000) # [cite: 106]
    :param reduction: the dimensional reduction (in adata.obsm) to use for clustering cells # Modified [cite: 106]
    :param dims: the number of dimensions from the reduction to use (default = 5) # [cite: 106]
    :param layer: Layer in adata to use for peak accessibility (e.g., 'counts', 'binary', 'tf-idf'). Uses adata.X if None. # Added
    :param random_seed: the random seed # [cite: 106]
    :return: the updated peak accessibility and graph distance matrices for metacells # Modified [cite: 106]
    """
    if reduction not in adata.obsm_keys():
        raise ValueError(f'Reduction "{reduction}" is not present. Available: {adata.obsm_keys()}') #[cite: 106]

    # Check if features (peaks) are valid
    if not all(f in adata.var_names for f in features):
         missing = [f for f in features if f not in adata.var_names]
         raise ValueError(f"Features (peaks) not found in adata.var_names: {missing[:5]}...")

    cell_embedding = adata.obsm[reduction][:, :dims] # [cite: 107]

    # Get peak accessibility matrix from specified layer or X
    if layer:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.")
        peak_accessibility = adata[:, features].layers[layer] # Renamed [cite: 107]
    else:
        peak_accessibility = adata[:, features].X # Renamed [cite: 107]

    # Handle sparse matrix if necessary for coarse_grain input
    if hasattr(peak_accessibility, 'toarray'):
         peak_accessibility = peak_accessibility.toarray()

    cg = coarse_grain(cell_embedding=cell_embedding,
                      peak_accessibility=peak_accessibility, # Renamed
                      graph_dist=graph_dist,
                      n=n, random_seed=random_seed) # [cite: 107]
    return cg