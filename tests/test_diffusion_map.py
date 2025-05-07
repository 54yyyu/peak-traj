import unittest
import numpy as np
from sklearn.metrics import pairwise_distances

# Import from peak_trajectory (though function itself is unchanged)
from peak_trajectory.diffusion_map import diffusion_map, get_symmetrized_affinity_matrix


class DiffusionMapTestCase(unittest.TestCase):
    # Test data remains the same as it tests the mathematical operation
    def test_diffusion_map(self):
        """ Test based on original R code comparison """
        ce = np.array([[0, 0], [2, 3], [3, 2], [5, 0], [0, 5]])
        gd = pairwise_distances(ce, metric='manhattan')

        diffu_emb, eigen_vals = diffusion_map(gd, k=3, n_ev=3, t=1)

        self.assertEqual(3, len(eigen_vals))
        # Eigenvalues should match
        np.testing.assert_array_almost_equal([1.0000000, 0.5219069, 0.3861848], eigen_vals, 6)

        # Eigenvectors can differ by sign, check absolute values
        expected_ev1_abs = np.abs([-0.292406, -0.292406, -0.292406, -0.292406, -0.292406])
        expected_ev2_abs = np.abs([6.666994e-17, -5.888965e-02, 5.888965e-02, 2.515596e-01, -2.515596e-01])
        expected_ev3_abs = np.abs([-0.15219742, 0.11845799, 0.11845799, -0.07091361, -0.07091361])

        np.testing.assert_array_almost_equal(expected_ev1_abs, np.abs(diffu_emb[:, 0]), 6)
        np.testing.assert_array_almost_equal(expected_ev2_abs, np.abs(diffu_emb[:, 1]), 6)
        np.testing.assert_array_almost_equal(expected_ev3_abs, np.abs(diffu_emb[:, 2]), 6)

    # Test data remains the same
    @staticmethod
    def test_get_symmetrized_affinity_matrix():
        ce = np.array([[0, 0], [2, 3], [3, 2]])
        gd = pairwise_distances(ce, metric='manhattan')

        affinity = get_symmetrized_affinity_matrix(gd, k=3)
        expected_affinity = np.array([
            [1., 0.367879, 0.367879],
            [0.367879, 1., 0.852144],
            [0.367879, 0.852144, 1.]
        ])
        np.testing.assert_almost_equal(expected_affinity, affinity, 6)


if __name__ == '__main__':
    unittest.main()