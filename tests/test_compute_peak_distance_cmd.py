import os
import unittest
from tempfile import TemporaryDirectory
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix

# Import adapted command line function
from peak_trajectory.compute_peak_distance_cmd import cal_ot_mat_cmd

# Renamed test class
class ComputePeakDistanceCmdTestCase(unittest.TestCase):
    # Reusing sample data structure, but interpreting as peak data
    # Cell-cell cost matrix
    cost_matrix = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 2, 1]])

    # Peak accessibility matrix (Cells x Peaks)
    peak_acc_matrix = np.array([
        [.3, .1, .6], # Cell 1 accessibilities for Peak A, B, C
        [.2, .3, .5], # Cell 2 ...
        [.6, .2, .2]  # Cell 3 ...
    ])

    # Expected peak-peak EMD matrix
    expected_peak_emd = np.array([
        [0.0, 0.8, 1.0], # EMD(A,A), EMD(A,B), EMD(A,C)
        [0.8, 0.0, 0.9], # EMD(B,A), EMD(B,B), EMD(B,C)
        [1.0, 0.9, 0.0]  # EMD(C,A), EMD(C,B), EMD(C,C)
    ])

    # Example peak pairs (0-indexed)
    peak_pairs = np.array([[0, 1], [1, 1], [2, 1]], dtype=int) # Pairs (A,B), (B,B), (C,B)

    # Renamed function
    def test_compute_peak_distance_cmd(self):
        with TemporaryDirectory() as d:
            # Define filenames used by the cmd function
            cost_file = "ot_cost.csv"
            accessibility_file = "peak_accessibility.mtx" # Renamed
            pairs_file = "peak_pairs.csv" # Renamed (though not used in this specific test)
            output_file = "emd_peaks.csv" # Renamed

            # Save input data
            np.savetxt(os.path.join(d, cost_file), self.cost_matrix, delimiter=',')
            # Save peak accessibility matrix (needs to be Peaks x Cells for mmwrite if using original structure)
            # The function expects Cells x Peaks, so save transpose? No, cmd reads it directly.
            # Let's save it as is (Cells x Peaks) and assume the cmd function handles it.
            # The original test saved gem.T, where gem was Cells x Genes. So save peak_acc_matrix.T
            sio.mmwrite(os.path.join(d, accessibility_file), coo_matrix(self.peak_acc_matrix.T))

            # Run the command line function simulation
            cal_ot_mat_cmd(
                path=d,
                cost_file=cost_file,
                accessibility_file=accessibility_file,
                pairs_file=pairs_file, # Pass default name even if file doesn't exist
                output_file=output_file,
                show_progress_bar=False
            )

            # Check the output
            res = np.loadtxt(os.path.join(d, output_file), delimiter=",")
            np.testing.assert_almost_equal(res, self.expected_peak_emd, decimal=6)

    # Renamed function
    def test_cal_ot_mat_peak_pairs(self):
        # Calculate expected result when only specific pairs are computed
        exp = np.full_like(self.expected_peak_emd, np.inf) # Default to inf for missing pairs
        np.fill_diagonal(exp, 0)
        exp[0, 1] = exp[1, 0] = self.expected_peak_emd[0, 1] # Pair (0, 1)
        # Pair (1, 1) is diagonal -> 0
        exp[2, 1] = exp[1, 2] = self.expected_peak_emd[2, 1] # Pair (2, 1)
        # Fill remaining inf with large value (as done in cal_ot_mat)
        max_finite = np.nanmax(exp[np.isfinite(exp)])
        fill_value = 1000 * max_finite if max_finite > 0 else 1.0
        exp[np.isinf(exp)] = fill_value


        with TemporaryDirectory() as d:
            # Define filenames
            cost_file = "ot_cost.csv"
            accessibility_file = "peak_accessibility.mtx"
            pairs_file = "peak_pairs.csv" # This file will be created
            output_file = "emd_peaks.csv"

            # Save input data
            np.savetxt(os.path.join(d, cost_file), self.cost_matrix, delimiter=',')
            sio.mmwrite(os.path.join(d, accessibility_file), coo_matrix(self.peak_acc_matrix.T))
            # Save the peak pairs file
            np.savetxt(os.path.join(d, pairs_file), self.peak_pairs, fmt='%d', delimiter=',')

            # Run the command line function simulation
            cal_ot_mat_cmd(
                path=d,
                cost_file=cost_file,
                accessibility_file=accessibility_file,
                pairs_file=pairs_file, # Pass the created pairs file
                output_file=output_file,
                show_progress_bar=False
            )

            # Check the output
            res = np.loadtxt(os.path.join(d, output_file), delimiter=",")
            np.testing.assert_almost_equal(res, exp, decimal=6)


if __name__ == '__main__':
    unittest.main()