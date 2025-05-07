import unittest
import numpy as np
import pandas as pd # Import pandas

# Import adapted functions and example data
from peak_trajectory.add_peak_bin_score import add_peak_bin_score
from tests.example_data import to_adata_peaks, peak_accessibility, peak_trajectories, peak_names

# Renamed test class
class AddPeakScoreTestCase(unittest.TestCase):
    # Renamed test function
    def test_add_peak_bin_score(self):
        # Create adata with cells as obs, peaks as vars
        adata = to_adata_peaks(peak_accessibility, var_names=peak_names)

        # Use the example peak_trajectories DataFrame
        # Note: example peak_trajectories has 3 peaks, adjust n_bins if needed
        test_peak_traj = peak_trajectories.copy()

        # Call the adapted function
        add_peak_bin_score(adata, test_peak_traj, trajectories=1, n_bins=2)

        # Check if the correct columns were added
        self.assertListEqual(['Trajectory1_peaks1', 'Trajectory1_peaks2'], adata.obs_keys())

        # Validate the calculated scores (adjust expected values based on example data)
        # Bin 1: Peak_A, Peak_B (2 peaks)
        # Cell 0: (1/2) * (peak_A>0) + (1/2)*(peak_B>0) = 0.5 * 1 + 0.5 * 0 = 0.5
        # Cell 1: 0.5 * 0 + 0.5 * 1 = 0.5
        # Cell 2: 0.5 * 1 + 0.5 * 0 = 0.5
        # Cell 3: 0.5 * 0 + 0.5 * 0 = 0.0
        # Cell 4: 0.5 * 1 + 0.5 * 1 = 1.0
        np.testing.assert_almost_equal([0.5, 0.5, 0.5, 0.0, 1.0], adata.obs['Trajectory1_peaks1'], decimal=5)

        # Bin 2: Peak_C (1 peak)
        # Cell 0: (1/1) * (peak_C>0) = 1.0 * 1 = 1.0
        # Cell 1: 1.0 * 0 = 0.0
        # Cell 2: 1.0 * 0 = 0.0
        # Cell 3: 1.0 * 0 = 0.0
        # Cell 4: 1.0 * 1 = 1.0
        np.testing.assert_almost_equal([1.0, 0.0, 0.0, 0.0, 1.0], adata.obs['Trajectory1_peaks2'], decimal=5)

if __name__ == '__main__':
    unittest.main()