import logging
from typing import Union, Optional, List # Added List

from scipy.stats import rankdata
import numpy as np
import pandas as pd

# Assuming these utils are moved or path adjusted
# Renamed import if diffusion_map remains separate or adjust path
from peak_trajectory.diffusion_map import diffusion_map, get_symmetrized_affinity_matrix
from peak_trajectory.util.input_validation import validate_matrix

logger = logging.getLogger()


# Renamed from get_gene_embedding
def get_peak_embedding(
        dist_mat: np.array, # Peak-peak distance matrix
        k: int = 10,
        sigma: Union[float, np.array, list] = None,
        n_ev: int = 30, # [cite: 120]
        t: int = 1, # [cite: 120]
) -> tuple[np.array, np.array]:
    """
    Get the diffusion embedding of peaks based on the peak-peak Wasserstein distance matrix.
    (Adapted from get_gene_embedding)

    :param dist_mat: Peak-peak Wasserstein distance matrix (symmetric) # Modified [cite: 121]
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor # [cite: 121]
    :param sigma: Fixed kernel bandwidth, `sigma` will be ignored if `K` is specified # [cite: 121]
    :param n_ev: Number of leading eigenvectors to export # [cite: 121]
    :param t: Number of diffusion times # [cite: 121]
    :return: the peak diffusion embedding and the eigenvalues # Modified
    """
    validate_matrix(dist_mat, square=True) # [cite: 121]

    k = min(k, dist_mat.shape[0])
    n_ev = min(n_ev + 1, dist_mat.shape[0]) # Keep n_ev+1 for DM calculation

    # Diffusion map calculation remains the same
    diffu_emb, eigen_vals = diffusion_map(dist_mat=dist_mat, k=k, sigma=sigma, n_ev=n_ev, t=t) # [cite: 122]
    # Exclude the first trivial eigenvector
    diffu_emb = diffu_emb[:, 1:n_ev + 1] # [cite: 122]
    eigen_vals = eigen_vals[1:n_ev + 1] # [cite: 122]
    return diffu_emb, eigen_vals


def get_randow_walk_matrix( # Function name typo remains from source
        dist_mat: np.array, # Peak-peak distance matrix
        k: int = 10,
) -> np.array:
    """
    Convert a distance matrix into a random-walk matrix based on adaptive Gaussian kernel.
    (Works for peak-peak distance matrix)

    :param dist_mat: Precomputed distance matrix (symmetric, peak-peak) # Modified [cite: 123]
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor # [cite: 123]
    :return: Random-walk matrix
    """
    validate_matrix(dist_mat, square=True) # [cite: 123]

    # Calculation remains the same
    affinity_matrix_symm = get_symmetrized_affinity_matrix(dist_mat=dist_mat, k=k) # [cite: 123]
    row_sums = affinity_matrix_symm.sum(axis=1)
    # Avoid division by zero if a row sum is zero (isolated peak)
    row_sums[row_sums == 0] = 1.0
    normalized_vec = 1 / row_sums
    affinity_matrix_norm = (affinity_matrix_symm * normalized_vec[:, None]) #[cite: 123]

    return affinity_matrix_norm


# Renamed from get_gene_pseudoorder
def get_peak_pseudoorder(
        dist_mat: np.array, # Peak-peak distance matrix
        subset: list[int], # Indices of peaks in the trajectory
        max_id: Optional[int] = None, # Index of the terminal peak
) -> np.array:
    """
    Order peaks along a given trajectory.
    (Adapted from get_gene_pseudoorder)

    :param dist_mat: Peak-peak Wasserstein distance matrix (symmetric) # Modified [cite: 124]
    :param subset: Indices of peaks in a given trajectory # Modified [cite: 124]
    :param max_id: Index of the terminal peak # Modified [cite: 124]
    :return: The pseudoorder for all peaks (0 for peaks not in the subset) # Modified
    """
    validate_matrix(dist_mat, square=True) # [cite: 124]

    if not subset: # Handle empty trajectory case
         return np.zeros(dist_mat.shape[0])

    emd = dist_mat[np.ix_(subset, subset)] # Use np.ix_ for safe indexing
    # Run diffusion map on the subset of peaks
    dm_emb, _ = diffusion_map(emd) # [cite: 124]

    # Use the first non-trivial eigenvector for ordering
    pseudoorder = rankdata(dm_emb[:, 1]) # [cite: 124]

    if max_id is not None and max_id in subset:
        n = len(subset)
        try:
            max_id_subset_index = subset.index(max_id)
            if 2 * pseudoorder[max_id_subset_index] < n: # Correct indexing
                pseudoorder = n + 1 - pseudoorder # [cite: 125]
        except ValueError:
             # max_id was not found in subset, though it should be if passed correctly
             logger.warning(f"max_id {max_id} not found in trajectory subset.")


    pseudoorder_all = np.zeros(dist_mat.shape[0]) # [cite: 125]
    pseudoorder_all[subset] = pseudoorder #[cite: 125]
    return pseudoorder_all


# Renamed from extract_gene_trajectory
def extract_peak_trajectory(
        peak_embedding: pd.DataFrame, # Renamed
        dist_mat: np.array, # Peak-peak distance matrix
        peak_names: List[str], # Renamed [cite: 126]
        t_list: Union[float, np.array, list], # [cite: 126]
        n: Optional[int] = None, # [cite: 127]
        dims: int = 5, # [cite: 126]
        k: int = 10, # [cite: 126]
        quantile: float = 0.02, # [cite: 126, 128]
        other: str = 'Other', # [cite: 129]
) -> pd.DataFrame:
    """
    Extract peak trajectories.
    (Adapted from extract_gene_trajectory)

    :param peak_embedding: Peak embedding (e.g., from get_peak_embedding) # Renamed [cite: 126]
    :param dist_mat: Peak-peak Wasserstein distance matrix (symmetric) # Renamed [cite: 126]
    :param peak_names: List of peak names/identifiers corresponding to rows/cols of dist_mat # Renamed
    :param t_list: Number of diffusion times to retrieve each trajectory # [cite: 127]
    :param n: Number of peak trajectories to retrieve. Will be set to the length of t_list if None. # Modified [cite: 127]
    :param dims: Dimensions of peak embedding to use to identify terminal peaks (extrema) # Modified [cite: 127]
    :param k: Adaptive kernel bandwidth for random walk matrix construction # Modified [cite: 127]
    :param quantile: Thresholding parameter (relative to max diffused value) to extract peaks for each trajectory. [cite: 127, 128]
    :param other: Label for peaks not in a trajectory. [cite: 129]
    :return: A data frame indicating peak trajectories and peak ordering along each trajectory # Modified [cite: 129]
    """
    validate_matrix(dist_mat, square=True) # [cite: 129]
    if peak_embedding.shape[0] != dist_mat.shape[0]:
         raise ValueError("peak_embedding rows must match dist_mat dimensions.")
    if len(peak_names) != dist_mat.shape[0]:
         raise ValueError("Length of peak_names must match dist_mat dimensions.")

    if np.isscalar(t_list):
        if n is None:
            raise ValueError(f'n should be specified if t_list is a number: {t_list}') # [cite: 130]
        t_list = np.full(n, t_list)
    elif n is None:
        n = len(t_list) # [cite: 130]
    if n != len(t_list):
        raise ValueError(f't_list ({t_list}) should have the same dimension as n ({n})') # [cite: 130]

    # Calculate distance from origin in the peak embedding space
    dist_to_origin = np.sqrt((peak_embedding[:, :dims] ** 2).sum(axis=1)) # [cite: 130]
    df = pd.DataFrame(peak_embedding[:, :dims], columns=[f'DM_{i + 1}' for i in range(dims)],
                      index=peak_names).assign(selected=other) # Use peak_names as index [cite: 130]
    n_peaks = peak_embedding.shape[0] # Renamed [cite: 130]

    diffusion_mat = get_randow_walk_matrix(dist_mat, k=k) # [cite: 130]

    for i in range(n):
        if sum(df.selected == other) == 0:
            logger.warning(f"Early stop reached. No remaining peaks to assign. {i} peak trajectories were retrieved.") # Modified [cite: 131]
            break

        # Find terminal peak (furthest from origin among unassigned peaks)
        dist_to_origin[df.selected != other] = -np.inf # [cite: 131]
        if np.all(np.isinf(dist_to_origin)): # Check if all remaining peaks are assigned (should not happen if sum > 0)
             logger.warning(f"Stopping early: all peaks seem assigned or distance calculation failed at step {i}.")
             break
        seed_idx = np.argmax(dist_to_origin) # [cite: 131]
        logger.info(f'Generating trajectory {i+1} from peak: {peak_names[seed_idx]}') # Modified

        # Simulate random walk starting from the seed peak
        seed_diffused = np.zeros(n_peaks) # Renamed
        seed_diffused[seed_idx] = 1 # [cite: 131]

        # Apply diffusion matrix t_list[i] times
        if t_list[i] > 0: # Check for t=0 case
             current_diffusion = diffusion_mat
             # Efficiently calculate matrix power using repeated squaring if t is large,
             # but simple iteration is fine for typical small t values.
             # Using np.linalg.matrix_power might be numerically unstable for stochastic matrices.
             # Iterative multiplication is safer.
             if t_list[i] > 1:
                  powered_diffusion_mat = np.linalg.matrix_power(diffusion_mat, t_list[i]) # Use matrix power
             else:
                  powered_diffusion_mat = diffusion_mat
             seed_diffused = powered_diffusion_mat[seed_idx, :] # More direct way to get diffused state from seed
             # Or iterative approach (safer for stability):
             # seed_diffused_iter = np.zeros(n_peaks)
             # seed_diffused_iter[seed_idx] = 1
             # for _ in range(t_list[i]):
             #     seed_diffused_iter = diffusion_mat @ seed_diffused_iter
             # seed_diffused = seed_diffused_iter

        # Assign peaks to trajectory based on diffusion value
        cutoff = np.max(seed_diffused) * quantile # [cite: 132]
        if cutoff <= 0: # Handle cases where diffusion leads to all zeros or max is zero
             logger.warning(f"Cutoff is zero or negative for trajectory {i+1}. Skipping assignment.")
             df[f'Pseudoorder-{i + 1}'] = 0 # Assign 0 pseudoorder
             continue

        trajectory_label = f'Trajectory-{i + 1}' # [cite: 132]
        # Assign peaks above cutoff AND currently unassigned ('Other')
        df.loc[(seed_diffused > cutoff) & (df.selected == other), 'selected'] = trajectory_label # [cite: 132]

        # Calculate pseudoorder for the newly assigned peaks
        current_trajectory_indices = list(np.where(df.selected == trajectory_label)[0])
        df[f'Pseudoorder-{i + 1}'] = get_peak_pseudoorder(dist_mat=dist_mat,
                                                          subset=current_trajectory_indices, # Pass indices [cite: 133]
                                                          max_id=seed_idx) # [cite: 133]
    return df