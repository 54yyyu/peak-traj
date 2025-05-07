import logging
from concurrent.futures import ThreadPoolExecutor # Keep ThreadPoolExecutor as per original note
import os
import time
from multiprocessing.managers import SharedMemoryManager # [cite: 140]
from typing import Optional, Sized, Union # Added Union

import numpy as np # [cite: 134]
import ot # [cite: 134]
from tqdm import tqdm # [cite: 134]

# Assuming these utils are moved or path adjusted
from peak_trajectory.util.input_validation import validate_matrix # [cite: 134]
from peak_trajectory.util.shared_array import SharedArray, PartialStarApply # [cite: 134]

logger = logging.getLogger()
_DEFAULT_NUMITERMAX = 50000 # [cite: 134]


# Renamed from cal_ot_mat
def cal_ot_mat(
        ot_cost: np.array, # Cell-cell cost matrix
        peak_acc: np.array, # Renamed from gene_expr (cells x peaks)
        feature_pairs: Optional[Sized] = None, # Renamed from gene_pairs (peak pairs)
        num_iter_max=_DEFAULT_NUMITERMAX, # [cite: 135]
        show_progress_bar=True, # [cite: 135]
        processes: Optional[int] = None, # [cite: 135]
) -> np.array:
    """
    Calculate the peak-peak earth mover distance matrix using shared memory.
    (Adapted from cal_ot_mat for genes)
    Note that this step is computationally expensive [cite: 136]
    and will be performed in parallel. [cite: 136]

    :param ot_cost: the cell-cell cost matrix [cite: 137]
    :param peak_acc: the peak accessibility matrix (cells x peaks) # Renamed [cite: 137]
    :param feature_pairs: only compute the distance for the given peak pairs (0-indexed) (default: None). # Renamed [cite: 137]
                          the distance entry for missing pairs will be set to a large value. # Modified [cite: 138]
    :param num_iter_max: the max number of iterations when computing the distance (see ot.emd2) # [cite: 138]
    :param show_progress_bar: shows a progress bar while running the computation (default: True) # [cite: 138]
    :param processes:the number of processes to use (defaults to the number of CPUs available) # [cite: 138]
    :return: the peak-peak distance matrix # Modified [cite: 138]
    """
    # processes = int(processes) if isinstance(processes, (int, float)) else os.cpu_count() # Original had float check[cite: 138], simplified
    if processes is None:
        processes = os.cpu_count()
    else:
        processes = int(processes)


    validate_matrix(peak_acc, obj_name='Peak Accessibility Matrix', min_value=0) # Renamed [cite: 138]
    ncells, nfeatures = peak_acc.shape # Renamed nfeatures (is npeaks)
    validate_matrix(ot_cost, obj_name='Cost Matrix', shape=(ncells, ncells), min_value=0) # [cite: 139]

    if show_progress_bar:
        logger.info(f'Computing peak-peak emd distance...') # Modified

    # Renamed variables
    if feature_pairs is None:
        pairs = ((i, j) for i in range(0, nfeatures - 1) for j in range(i + 1, nfeatures))
        npairs = (nfeatures * (nfeatures - 1)) // 2 # [cite: 139]
    else:
        # Validate feature_pairs format if necessary (e.g., list of tuples/lists)
        pairs = feature_pairs
        npairs = len(feature_pairs) # [cite: 139]

    # Use np.inf as fill value initially for missing pairs
    emd_mat = np.full((nfeatures, nfeatures), fill_value=np.inf) # Renamed, changed fill [cite: 139]

    with SharedMemoryManager() as manager: # [cite: 140]
        start_time = time.perf_counter() # [cite: 140]

        # Use ThreadPoolExecutor as per original implementation note
        with ThreadPoolExecutor(max_workers=processes) as pool: # [cite: 140]
            # prepare shared environment
            cost = SharedArray.copy(manager, np.asarray(ot_cost)) # [cite: 140]
            feat_matrix = SharedArray.copy(manager, np.asarray(peak_acc)) # Renamed gexp to feat_matrix [cite: 140]
            # Pass feat_matrix to the worker function
            f = PartialStarApply(_cal_ot_peaks, cost, feat_matrix) # Renamed worker func [cite: 141]

            # execute tasks and process results
            result_generator = pool.map(f,  ((num_iter_max, i, j) for i, j in pairs)) # [cite: 141]
            if show_progress_bar:
                result_generator = tqdm(result_generator, total=npairs, position=0, leave=True) # [cite: 141]
            for d, i, j in result_generator:
                emd_mat[i, j] = emd_mat[j, i] = d # [cite: 142]

        finish_time = time.perf_counter() # [cite: 142]
        if show_progress_bar:
            logger.info("EMD computation finished in {} seconds - using {} workers".format(finish_time - start_time, processes)) # Modified [cite: 142]

        # Set diagonal to 0
        np.fill_diagonal(emd_mat, 0) # [cite: 142]

        # Replace remaining np.inf (missing pairs if feature_pairs was used) with a large value relative to computed distances
        if np.any(np.isinf(emd_mat)):
             max_finite_dist = np.nanmax(emd_mat[np.isfinite(emd_mat)])
             # Use a large multiple, or handle differently if needed
             fill_value = 1000 * max_finite_dist if max_finite_dist > 0 else 1.0 # Avoid 0 fill if all distances are 0
             emd_mat[np.isinf(emd_mat)] = fill_value # [cite: 142]

        # Final check for NaNs just in case, replace with fill_value
        if np.any(np.isnan(emd_mat)):
            logger.warning("NaN values found in EMD matrix after computation. Replacing with large value.")
            max_finite_dist = np.nanmax(emd_mat) # Recalculate max ignoring NaN
            fill_value = 1000 * max_finite_dist if max_finite_dist > 0 else 1.0
            np.nan_to_num(emd_mat, nan=fill_value, copy=False)

        return emd_mat


# Renamed from _cal_ot
def _cal_ot_peaks(ot_cost_matrix: SharedArray, peak_acc_matrix: SharedArray, num_iter_max: int, i: int, j: int):
    """
    Computes the EMD between peak i and peak j using shared memory arrays.
    (Adapted from _cal_ot for genes) # Modified
    """
    # Unpack shared memory arrays
    # ot_cost_matrix = ot_cost_matrix_sh.as_array()
    # peak_acc_matrix = peak_acc_matrix_sh.as_array()

    # Extract accessibility vectors for peak i and peak j
    peak_i = peak_acc_matrix[:, i].copy() # Renamed, use .copy() for safety [cite: 143]
    peak_j = peak_acc_matrix[:, j].copy() # Renamed, use .copy() for safety [cite: 143]

    # Normalize to create distributions (handle sum=0 case)
    sum_i = np.sum(peak_i)
    sum_j = np.sum(peak_j)

    if sum_i == 0 or sum_j == 0:
        # If one peak has zero accessibility everywhere, distance is ill-defined or infinite.
        # Return a large value or np.inf depending on desired behavior. POT might error.
        # Let's return np.inf and handle it in the main function.
        return np.inf, i, j

    peak_i /= sum_i # [cite: 143]
    peak_j /= sum_j # [cite: 143]

    # Find non-zero elements for EMD calculation (cells where peaks are accessible)
    nz_i = np.nonzero(peak_i)[0]
    nz_j = np.nonzero(peak_j)[0]

    if len(nz_i) == 0 or len(nz_j) == 0:
        # Should be caught by sum check above, but as a safeguard
        return np.inf, i, j

    # Get subset of cost matrix corresponding to accessible cells
    cost_subset = ot_cost_matrix[np.ix_(nz_i, nz_j)] # Use np.ix_ for safe slicing

    # Compute EMD using POT
    try:
        emd_dist = ot.emd2(peak_i[nz_i], # Pass only non-zero values
                           peak_j[nz_j], # Pass only non-zero values
                           cost_subset,
                           numItermax=num_iter_max) # [cite: 144]
    except Exception as e:
        logger.error(f"Error computing EMD between peaks {i} and {j}: {e}")
        emd_dist = np.inf # Return inf on error

    # return the generated value
    return emd_dist, i, j # [cite: 144]