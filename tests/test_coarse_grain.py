import unittest
import numpy as np
import pandas as pd # Import pandas
from sklearn.metrics import pairwise_distances

# Import adapted functions and example data
from peak_trajectory.coarse_grain import coarse_grain, select_top_peaks, coarse_grain_adata_peaks
from tests.example_data import random_adata_peaks # Use peak version

class CoarseGrainTestCase(unittest.TestCase):
    # Test coarse_grain directly (logic is independent of feature type name)
    @staticmethod
    def test_cg_logic():
        """ Test the core coarse_grain calculation logic. """
        ce = np.array([[0, 0], [2, 3], [3, 2], [5, 0], [0, 5]]) # Cell embedding
        # Feature matrix (can be peaks or genes)
        feature_matrix = np.array([[2, 0, 1], [0, 3, 0], [3, 0, 0], [0, 0, 0], [1, 1, 1]])
        n_metacells = 4
        gd = pairwise_distances(ce, metric='manhattan')
        # Predefined cluster assignment for reproducibility
        cluster = np.array([2, 3, 3, 0, 1]) # cluster indices should be 0 to n_metacells-1

        # Call the function
        feature_matrix_updated, graph_dist_updated = coarse_grain(
            cell_embedding=ce,
            peak_accessibility=feature_matrix, # Pass the feature matrix here
            graph_dist=gd,
            n=n_metacells,
            cluster=cluster
        )

        # Expected results based on the R code comments in original test
        expected_feature_matrix = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 1], [3, 3, 0]])
        expected_graph_dist = np.array([[0, 10, 5, 5], [10, 0, 5, 5], [5, 5, 0, 5], [5, 5, 5, 1]])

        np.testing.assert_array_equal(expected_feature_matrix, feature_matrix_updated)
        np.testing.assert_array_almost_equal(expected_graph_dist, graph_dist_updated, decimal=5) # Use almost equal for distances

    # Test the placeholder select_top_peaks function
    def test_select_top_peaks(self):
        adata = random_adata_peaks(shape=(100, 500), seed=123) # Use smaller number of peaks

        # Add dummy variance data needed by the placeholder
        adata.var['variance'] = np.random.rand(adata.n_vars)
        # Add dummy accessibility filter data
        perc_accessible = np.array(adata.X.astype(bool).sum(axis=0)).flatten() / adata.n_obs
        adata.var['peak_accessible_filt'] = (perc_accessible > 0.01) & (perc_accessible < 0.9)


        peaks = select_top_peaks(adata, layer='counts', n_variable_peaks=50)

        self.assertIsInstance(peaks, np.ndarray)
        self.assertTrue(len(peaks) <= 50)
        self.assertTrue(len(peaks) > 0) # Expect some peaks to be selected
        self.assertTrue(all(p in adata.var_names for p in peaks))

    # Test the coarse_grain_adata_peaks wrapper
    def test_coarse_grain_adata_peaks(self):
        adata = random_adata_peaks(shape=(100, 500), seed=42)
        # Dummy graph distance
        graph_dist = np.random.rand(adata.n_obs, adata.n_obs)
        np.fill_diagonal(graph_dist, 0)
        # Dummy peak selection
        selected_peaks = adata.var_names[:50].tolist() # Select first 50 peaks

        n_metacells = 20
        # Call the adapted wrapper function
        peak_acc_cg, graph_dist_cg = coarse_grain_adata_peaks(
            adata,
            graph_dist=graph_dist,
            features=selected_peaks,
            n=n_metacells,
            reduction="X_pca", # Use the dummy PCA embedding
            dims=5,
            layer='counts' # Use the counts layer
        )

        self.assertEqual(peak_acc_cg.shape, (n_metacells, len(selected_peaks)))
        self.assertEqual(graph_dist_cg.shape, (n_metacells, n_metacells))
        self.assertTrue(np.all(peak_acc_cg >= 0)) # Accessibility should be non-negative
        self.assertTrue(np.all(np.diag(graph_dist_cg) == 0)) # Diagonal distance should be 0

if __name__ == '__main__':
    unittest.main()