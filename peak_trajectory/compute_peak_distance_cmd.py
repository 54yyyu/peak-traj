import argparse
import os
import scipy.io as sio
from scipy.sparse import issparse
import numpy as np # [cite: 108]

# Assuming this util is moved or path adjusted
# Renamed import
from peak_trajectory.peak_distance_shared import _DEFAULT_NUMITERMAX, cal_ot_mat as cal_ot_mat_from_numpy


def parse_args():
    # Renamed description and help texts
    parser = argparse.ArgumentParser(description='Calculate the peak-peak earth mover distance matrix')
    parser.add_argument('--path', type=os.path.abspath, required=True,
                        help='The path containing the OT cost matrix (ot_cost.csv)' +
                             ' and the peak accessibility matrix (peak_accessibility.mtx)') # Renamed
    parser.add_argument('--num_iter_max', type=int, default=_DEFAULT_NUMITERMAX, # [cite: 109]
                        help='The max number of iterations when computing the distance (see ot.emd2)')
    parser.add_argument('--show_progress_bar', type=bool, default=True, # [cite: 109]
                        help='Shows a progress bar while running the computation (default: True)')
    parser.add_argument('--processes', type=int, required=False, default=None, # [cite: 110]
                        help='The number of processes to use (defaults to the number of CPUs available)')
    parser.add_argument('--pairs_file', type=str, default="peak_pairs.csv",
                        help='Optional file name for precomputed peak pairs (default: peak_pairs.csv)') # Renamed
    parser.add_argument('--cost_file', type=str, default="ot_cost.csv",
                        help='File name for the OT cost matrix (default: ot_cost.csv)')
    parser.add_argument('--accessibility_file', type=str, default="peak_accessibility.mtx",
                        help='File name for the peak accessibility matrix (default: peak_accessibility.mtx)') # Renamed
    parser.add_argument('--output_file', type=str, default="emd_peaks.csv",
                        help='File name for the output peak EMD matrix (default: emd_peaks.csv)') # Renamed

    return parser.parse_args()


# Renamed function and parameters
def cal_ot_mat_cmd(
    path: str,
    cost_file: str,
    accessibility_file: str,
    pairs_file: str,
    output_file: str,
    num_iter_max=_DEFAULT_NUMITERMAX,
    show_progress_bar=True,
    processes: int = None,
) -> None:
    """
    Calculate the earth mover distance matrix between peaks using command-line inputs.
    Note that this step is computationally expensive [cite: 111]
    and will be performed in parallel. [cite: 111]

    :param path: path to the folder where the cost matrix and peak accessibility matrix are saved # Modified [cite: 112]
    :param cost_file: Name of the OT cost matrix file (e.g., "ot_cost.csv") # Added
    :param accessibility_file: Name of the peak accessibility matrix file (e.g., "peak_accessibility.mtx") # Added/Renamed
    :param pairs_file: Name of the optional peak pairs file (e.g., "peak_pairs.csv") # Added/Renamed
    :param output_file: Name for the output EMD matrix file (e.g., "emd_peaks.csv") # Added/Renamed
    :param num_iter_max: the max number of iterations when computing the distance (see ot.emd2) # [cite: 112]
    :param show_progress_bar: shows a progress bar while running the computation (default: True) # [cite: 112]
    :param processes:the number of processes to use (defaults to the number of CPUs available) # [cite: 112]
    """
    ot_cost = np.loadtxt(os.path.join(path, cost_file), delimiter=",")
    peak_acc = sio.mmread(os.path.join(path, accessibility_file)) # Renamed [cite: 113]
    if issparse(peak_acc): # Renamed
        peak_acc = peak_acc.todense() # Renamed [cite: 113]

    peak_pairs_file_path = os.path.join(path, pairs_file) # Renamed
    # Renamed variable
    peak_pairs = np.loadtxt(peak_pairs_file_path, delimiter=",").astype(int) if os.path.isfile(peak_pairs_file_path) else None # [cite: 113]

    # Renamed variables passed to the core function
    emd_mat_peaks = cal_ot_mat_from_numpy(ot_cost=ot_cost, peak_acc=peak_acc, # Renamed peak_acc
                                          feature_pairs=peak_pairs, # Renamed feature_pairs
                                          num_iter_max=num_iter_max,
                                          show_progress_bar=show_progress_bar,
                                          processes=processes) # [cite: 114]

    np.savetxt(os.path.join(path, output_file), emd_mat_peaks, delimiter=",") # Renamed output file


if __name__ == '__main__':
    args = parse_args()

    # Renamed function call and arguments
    cal_ot_mat_cmd(path=args.path,
                   cost_file=args.cost_file,
                   accessibility_file=args.accessibility_file,
                   pairs_file=args.pairs_file,
                   output_file=args.output_file,
                   num_iter_max=args.num_iter_max,
                   show_progress_bar=args.show_progress_bar,
                   processes=args.processes) # [cite: 114]