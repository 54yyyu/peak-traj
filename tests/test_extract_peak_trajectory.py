import unittest
import numpy as np
import pandas as pd

# Import adapted functions
from peak_trajectory.extract_peak_trajectory import (
    get_peak_embedding,
    get_randow_walk_matrix, # Typo from original preserved
    get_peak_pseudoorder,
    extract_peak_trajectory
)
# Import adapted example data
from tests.example_data import peak_names # Use peak_names


# Renamed test class
class ExtractPeakTrajectoryTestCase(unittest.TestCase):
    # Using the same distance matrix structure, interpreting rows/cols as peaks
    peak_dist_mat = np.array([
        [0.000000, 4.269544, 4.414329, 7.247308, 8.027305],
        [4.269544, 0.000000, 6.927429, 6.761171, 8.801408],
        [4.414329, 6.927429, 0.000000, 6.531824, 6.798761],
        [7.247308, 6.761171, 6.531824, 0.000000, 5.766003],
        [8.027305, 8.801408, 6.798761, 5.766003, 0.000000]])

    # Example peak embedding (Peaks x Dims) - structure matches original gem
    peak_embedding_mat = np.array([
        [0.17043479, 0.04309062, 0.02332959],
        [0.11273428, -0.10420329, 0.04087336],
        [0.02461065, 0.09818398, -0.06456570],
        [-0.12521597, -0.08515901, -0.07519815],
        [-0.20356317, 0.04675745, 0.08842471]])

    # Use peak names corresponding to the matrix dimensions
    test_peak_names = [f"Peak_{i}" for i in range(5)]

    # Expected trajectory result structure for testing extract_peak_trajectory
    expected_peak_traj_df = pd.DataFrame({
        'DM_1': [0.170435, 0.112734, 0.024611, -0.125216, -0.203563],
        'DM_2': [0.043091, -0.104203, 0.098184, -0.085159, 0.046757], # Only first 2 dims used in test call
        'selected': ['Trajectory-1'] * 5,
        'Pseudoorder-1': [1.0, 2.0, 3.0, 4.0, 5.0], # float due to rankdata
    }, index=test_peak_names)


    # Renamed test function
    def test_get_peak_embedding(self):
        # Test the embedding function
        diffu_emb, eigen_vals = get_peak_embedding(self.peak_dist_mat, k=3, n_ev=3)
        self.assertEqual(3, len(eigen_vals))
        # Eigenvalues remain the same
        np.testing.assert_array_almost_equal([0.4765176, 0.2777494, 0.2150586], eigen_vals, 6)

        # Check absolute values of eigenvectors
        np.testing.assert_array_almost_equal(np.abs(self.peak_embedding_mat[:, 0]), np.abs(diffu_emb[:, 0]), 6)
        np.testing.assert_array_almost_equal(np.abs(self.peak_embedding_mat[:, 1]), np.abs(diffu_emb[:, 1]), 6)
        np.testing.assert_array_almost_equal(np.abs(self.peak_embedding_mat[:, 2]), np.abs(diffu_emb[:, 2]), 6)

    # Test random walk matrix (logic unchanged)
    def test_get_random_walk_matrix(self):
        rw = np.array([
            [0.51517102, 0.18952083, 0.1832053, 0.06750703, 0.04459585],
            [0.22037146, 0.59903173, 0.0470527, 0.10012794, 0.03341618],
            [0.19758186, 0.04364105, 0.5555978, 0.10809274, 0.09508657],
            [0.07042714, 0.08983549, 0.1045631, 0.53745545, 0.19771881],
            [0.05148485, 0.03317748, 0.1017877, 0.21879730, 0.59475272],
        ])
        np.testing.assert_almost_equal(rw, get_randow_walk_matrix(self.peak_dist_mat, k=2), 6)

    # Renamed test function
    def test_get_peak_pseudoorder(self):
        # Test pseudo-ordering logic
        # Assuming peak at index 4 is terminal
        np.testing.assert_array_equal([1., 2., 3., 4., 5.], get_peak_pseudoorder(self.peak_dist_mat, list(range(5)), max_id=4))
        # Assuming peak at index 0 is terminal (should reverse order)
        np.testing.assert_array_equal([5., 4., 3., 2., 1.], get_peak_pseudoorder(self.peak_dist_mat, list(range(5)), max_id=0))
        # Test subset
        subset_indices = [1, 3, 4] # Peaks Peak_1, Peak_3, Peak_4
        # Assuming max_id=1 (Peak_1) - its rank should be highest in the subset's order
        expected_subset_order = np.zeros(5)
        expected_subset_order[[1, 3, 4]] = [3., 2., 1.] # Reversed order based on DM1 of subset
        np.testing.assert_array_equal(expected_subset_order, get_peak_pseudoorder(self.peak_dist_mat, subset_indices, max_id=1))

    # Renamed test function
    def test_extract_peak_trajectory(self):
        # Test the main trajectory extraction function
        gt = extract_peak_trajectory(
             peak_embedding=self.peak_embedding_mat, # Pass the numpy array
             dist_mat=self.peak_dist_mat,
             peak_names=self.test_peak_names, # Pass list of names
             n=1,
             t_list=[3],
             dims=2 # Use only first 2 dims of embedding
        )

        # Compare structure and values with expected output
        pd.testing.assert_frame_equal(
            self.expected_peak_traj_df[['DM_1', 'DM_2', 'selected', 'Pseudoorder-1']],
            gt[['DM_1', 'DM_2', 'selected', 'Pseudoorder-1']],
            check_exact=False, # Allow for float precision differences
            atol=1e-6
        )

if __name__ == '__main__':
    unittest.main()