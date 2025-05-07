import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

# Renamed from gene_expression
peak_accessibility = np.array([
    [2, 0, 1],  # Cells x Peaks
    [0, 3, 0],
    [3, 0, 0],
    [0, 0, 0],
    [1, 1, 1]
])

# This represents cell embedding (e.g., LSI, PCA, or DM of cells)
# Renamed from diffusion_map (as it was ambiguous - representing cell DM embedding)
cell_embedding_dm = np.array([
    [-0.04764353, 0.00940602, -0.00042641, -0.00158765],
    [0.11828658, 0.04134494, -0.00401907, -0.00012575],
    [-0.06615087, 0.03891922, -0.00681983,  0.00119593],
    [0.01417467, -0.05808308, -0.01944058,  0.0002601],
    [0.00654969, -0.02393814,  0.02780562,  0.00040004]
])

# Cell-cell graph distance matrix
graph_distance = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 2, 1, 1],
    [1, 2, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
])

# Renamed from gene_names
peak_names = ["Peak_A", "Peak_B", "Peak_C"] # Example peak names

# Example peak trajectory result (DM embedding of peaks)
# Renamed from gene_trajectories
peak_trajectories = pd.DataFrame({
    'DM_1': [0.170435, 0.112734, -0.203563], # Example values for 3 peaks
    'DM_2': [0.043091, -0.104203, 0.046757],
    'selected': ['Trajectory-1']*3,
    'Pseudoorder-1': [1, 2, 3],
}, index=peak_names[:3]) # Use first 3 peak names for this example


# Renamed from example_adata
def example_adata_peaks() -> sc.AnnData:
    # Use peak_accessibility and peak_names
    return to_adata_peaks(peak_accessibility, var_names=peak_names)


# Renamed from to_adata
def to_adata_peaks(x: np.array, obs_names: list[str] = None, var_names: list[str] = None):
    # Ensure var_names corresponds to the columns (peaks)
    n_obs, n_vars = x.shape
    adata = sc.AnnData(csr_matrix(np.asarray(x, dtype=np.float32)))
    adata.obs_names = obs_names or [f"Cell_{i:d}" for i in range(n_obs)]
    adata.var_names = var_names or [f"Peak_{i:d}" for i in range(n_vars)]
    return adata


# Renamed from random_adata
def random_adata_peaks(shape=(100, 5000), seed=123) -> sc.AnnData: # Increased default peak number
    """
    Generate a reasonably sized Scanpy AnnData object with peak data.
    """
    prng = np.random.RandomState(seed)
    # Simulate peak data (e.g., poisson or binomial for accessibility)
    counts = csr_matrix(prng.poisson(0.5, size=shape), dtype=np.float32) # Lower lambda for sparsity

    adata = sc.AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Peak_{i:d}" for i in range(adata.n_vars)] # Use Peak prefix

    # Store raw counts if needed
    adata.raw = adata
    # Add a 'counts' layer often used in preprocessing steps
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy() # Use X directly

    # Add a dummy embedding needed for some tests
    adata.obsm['X_pca'] = np.random.rand(adata.n_obs, 10)

    return adata