import unittest
import numpy as np

# Import from peak_trajectory.util
from peak_trajectory.util.input_validation import validate_matrix, validate_not_none # Added validate_not_none

class InputValidationTestCase(unittest.TestCase):
    def test_validate_matrix(self):
        m = np.array([[1, 2], [3, 4]])

        # Valid cases
        validate_matrix(m, obj_name='TestMatrix', min_value=1, max_value=4, square=True, shape=(2, 2))
        validate_matrix(m, nrows=2, ncols=2)
        validate_matrix(m, min_size=1)

        # Invalid cases
        with self.assertRaisesRegex(ValueError, 'TestMatrix does not have 3 rows.*'):
            validate_matrix(m, obj_name='TestMatrix', nrows=3)
        with self.assertRaisesRegex(ValueError, 'TestMatrix does not have 8 columns.*'):
            validate_matrix(m, obj_name='TestMatrix', ncols=8)
        with self.assertRaisesRegex(ValueError, 'TestMatrix does not have shape \\(1, 1\\)'):
            validate_matrix(m, obj_name='TestMatrix', shape=(1, 1))
        with self.assertRaisesRegex(ValueError, 'TestMatrix does not have enough elements.*Min_size: 3.*'):
            validate_matrix(m, obj_name='TestMatrix', min_size=3)
        with self.assertRaisesRegex(ValueError, 'TestMatrix should not have values less than 5.*'):
            validate_matrix(m, obj_name='TestMatrix', min_value=5)
        with self.assertRaisesRegex(ValueError, 'TestMatrix should not have values greater than 1.*'):
            validate_matrix(m, obj_name='TestMatrix', max_value=1)
        with self.assertRaisesRegex(ValueError, 'TestMatrix is not a square matrix.*'):
             validate_matrix(np.array([[1,2,3],[4,5,6]]), obj_name='TestMatrix', square=True)
        with self.assertRaisesRegex(ValueError, 'TestMatrix is a square matrix.*'):
             validate_matrix(m, obj_name='TestMatrix', square=False)
        with self.assertRaisesRegex(ValueError, 'TestMatrix is not a matrix.*'):
             validate_matrix(np.array([1,2,3]), obj_name='TestMatrix')


    def test_validate_not_none(self):
         validate_not_none([1, 2], obj_name='TestList') # Should pass
         with self.assertRaisesRegex(ValueError, "TestNone is None"):
              validate_not_none(None, obj_name='TestNone')


if __name__ == '__main__':
    unittest.main()