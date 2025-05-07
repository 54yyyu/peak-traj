from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc

# Renamed from add_gene_bin_score
def add_peak_bin_score(
        adata: sc.AnnData,
        peak_trajectory: pd.DataFrame, # Renamed from gene_trajectory
        n_bins: int = 5,
        trajectories: int = 2,
        layer: Optional[str] = None,
        reverse: Optional[bool] = None, # [cite: 94]
        prefix: str = 'Trajectory',
) -> None:
    """
    Add peak bin score (adapted from gene bin score)

    :param adata: a scanpy AnnData object (cells x peaks)
    :param peak_trajectory: Peak trajectory data frame (result from extract_peak_trajectory) # Renamed
    :param n_bins: How many peak bins
    :param trajectories: Which peak trajectories to define peak bin score # Renamed
    :param layer: string Which layer to use for peak accessibility. Uses adata.X if empty # Modified [cite: 95]
    :param reverse: Whether to reverse the order of peaks along each trajectory # Renamed [cite: 95]
    :param prefix: String added to the names in the metadata
    """

    if layer is not None and layer not in adata.layers.keys():
        raise ValueError(f'Layer {layer} not found in adata. Available {list(adata.layers.keys())}')
    # Ensure the AnnData object has peaks in the var dimension
    # Add checks if necessary, e.g., adata.var_names contain peak identifiers

    for trajectory in range(1, trajectories + 1):
        trajectory_name = f'Pseudoorder-{trajectory}'

        peak_trajectory_reordered = peak_trajectory.sort_values(trajectory_name) # Renamed

        # Renamed 'genes' to 'peaks'
        peaks = peak_trajectory_reordered[peak_trajectory_reordered[trajectory_name] > 0].index.values # [cite: 96]
        if reverse:
            peaks = list(reversed(peaks)) # [cite: 96]
        step = len(peaks) / n_bins

        for i in range(n_bins):
            start = int(np.ceil(i * step))
            end = min(int(np.ceil((i + 1) * step)), len(peaks))
            peaks_subset = peaks[start:end] # Renamed [cite: 97]

            if not list(peaks_subset): # Handle empty bin case
                 adata.obs[f'{prefix}{trajectory}_peaks{i + 1}'] = 0.0
                 continue

            adata_subset = adata[:, peaks_subset]
            # Use peak accessibility matrix (e.g., counts, binary, TF-IDF)
            x = adata_subset.layers[layer] if layer else adata_subset.X
            # Score based on accessibility (e.g., fraction of accessible peaks in bin)
            # The definition "> 0" might need adjustment based on data type (e.g., for TF-IDF)
            normalized_accessibility = x > 0 # Renamed
            if hasattr(normalized_accessibility, 'toarray'): # Handle sparse matrix
                 normalized_accessibility = normalized_accessibility.toarray()
            # Ensure meta calculation handles potential division by zero if peaks_subset is empty (handled above)
            meta = np.squeeze(np.asarray(normalized_accessibility.sum(axis=1) / len(peaks_subset))) # [cite: 97] Renamed
            adata.obs[f'{prefix}{trajectory}_peaks{i + 1}'] = meta # Renamed