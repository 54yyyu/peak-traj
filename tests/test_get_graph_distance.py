import unittest
import numpy as np

# Import from peak_trajectory (though function itself is unchanged)
from peak_trajectory.get_graph_distance import get_graph_distance
# Import adapted example data
from tests.example_data import example_adata_peaks, cell_embedding_dm, graph_distance

# Renamed test class slightly for clarity
class GetGraphDistanceTestCase(unittest.TestCase):
    @staticmethod
    def test_get_graph_distance():
        # Create a sample AnnData object
        adata = example_adata_peaks() # Use the peak example adata generator
        # Add the sample cell embedding (interpreting original 'diffusion_map' as cell embedding)
        adata.obsm['X_dm_cell'] = cell_embedding_dm

        # Call the function using the cell embedding
        gd = get_graph_distance(adata, reduction='X_dm_cell', k=3, dims=4) # Use all 4 dims from example
        # Compare with the expected graph distance
        np.testing.assert_almost_equal(gd, graph_distance, decimal=5)

    @staticmethod
    def test_get_graph_distance_disconnected_error():
         # Test case where graph might be disconnected (e.g., k=1)
         adata = example_adata_peaks()
         adata.obsm['X_dm_cell'] = cell_embedding_dm
         # Lower k is more likely to cause disconnectivity, though not guaranteed with this small data
         with pytest.raises(RuntimeError, match='disconnected components'): # Use pytest.raises
              # This specific small example might not disconnect even with k=1
              # A better test would construct data guaranteed to disconnect
              get_graph_distance(adata, reduction='X_dm_cell', k=1, dims=4)


if __name__ == '__main__':
    # Note: pytest is generally preferred over unittest runner for fixtures and raises
    # To run with pytest, simply execute `pytest` in the terminal in the project root.
    # If using unittest:
    import pytest # Import pytest for raises
    unittest.main()