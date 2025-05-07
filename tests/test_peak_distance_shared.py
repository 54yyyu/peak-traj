import unittest
import numpy as np

# Import adapted function
from peak_trajectory.peak_distance_shared import cal_ot_mat

# Renamed test class
class PeakDistanceSharedTestCase(unittest.TestCase):
    # Reusing sample data structure, interpreting as peak data
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
    ]).T # Transpose to get Peaks x Cells if function expects that, or keep as Cells x Peaks
    # The function cal_ot_mat expects Cells x Peaks (ncells, nfeatures)
    peak_acc_matrix_cells_x_peaks = peak_acc_matrix.T

    # Expected peak-peak EMD matrix
    expected_peak_emd = np.array([
        [0.0, 0.8, 1.0], # EMD(A,A), EMD(A,B), EMD(A,C)
        [0.8, 0.0, 0.9], # EMD(B,A), EMD(B,B), EMD(B,C)
        [1.0, 0.9, 0.0]  # EMD(C,A), EMD(C,B), EMD(C,C)
    ])

    # Example peak pairs (0-indexed list of tuples)
    peak_pairs = [(0, 1), (1, 1), (2, 1)] # Pairs (A,B), (B,B), (C,B)

    # Renamed test function
    def test_peak_distance_shared(self):
        # Test the core calculation
        mt = cal_ot_mat(
            ot_cost=self.cost_matrix,
            peak_acc=self.peak_acc_matrix_cells_x_peaks, # Use Cells x Peaks format
            show_progress_bar=False
        )
        np.testing.assert_almost_equal(self.expected_peak_emd, mt, 6)

    # Renamed test function
    def test_peak_distance_input_validation(self):
        # Test input validation logic (should remain similar)
        # Mismatched cost matrix rows and peak accessibility rows (cells)
        with self.assertRaisesRegex(ValueError, 'Cost Matrix does not have shape.*'):
            cal_ot_mat(ot_cost=self.cost_matrix, peak_acc=np.ones(shape=(6, 3)), show_progress_bar=False)

        # Mismatched cost matrix cols and peak accessibility rows (cells)
        with self.assertRaisesRegex(ValueError, 'Cost Matrix does not have shape.*'):
            cal_ot_mat(ot_cost=np.ones(shape=(6, 3)), peak_acc=self.peak_acc_matrix_cells_x_peaks, show_progress_bar=False)

        # Negative values in peak accessibility matrix
        with self.assertRaisesRegex(ValueError, 'Peak Accessibility Matrix should not have values less than 0.*'):
             cal_ot_mat(ot_cost=self.cost_matrix, peak_acc=self.peak_acc_matrix_cells_x_peaks - 1, show_progress_bar=False)


    # Renamed test function
    def test_cal_ot_mat_peak_pairs(self):
        # Calculate expected result when only specific pairs are computed
        exp = np.full_like(self.expected_peak_emd, np.inf) # Default to inf for missing pairs
        np.fill_diagonal(exp, 0)
        exp[0, 1] = exp[1, 0] = self.expected_peak_emd[0, 1] # Pair (0, 1)
        # Pair (1, 1) is diagonal -> 0
        exp[2, 1] = exp[1, 2] = self.expected_peak_emd[2, 1] # Pair (2, 1)
        # Fill remaining inf with large value
        max_finite = np.nanmax(exp[np.isfinite(exp)])
        fill_value = 1000 * max_finite if max_finite > 0 else 1.0
        exp[np.isinf(exp)] = fill_value

        # Test calculation with specific pairs provided
        mt = cal_ot_mat(
            ot_cost=self.cost_matrix,
            peak_acc=self.peak_acc_matrix_cells_x_peaks,
            feature_pairs=self.peak_pairs, # Pass the list of peak pairs
            show_progress_bar=False
        )
        np.testing.assert_almost_equal(exp, mt, 6)


if __name__ == '__main__':
    unittest.main()