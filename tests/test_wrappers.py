import unittest

import ccec
import numpy as np
from numpy.testing import assert_array_equal

class TestEigency(unittest.TestCase):

    def test_function_w_mat_arg(self):
        a = 1.234
        x = np.array([1.1, 2.2, 3.3, 4.4])
        v = np.array([1.1, 2.2, 3.3, 4.4]) + 1.0
        ccec.function_w_mat_arg(x.reshape([2, 2]), v, a)
        # Shared memory test: Verify that first entry was set to 0 by C++ code.
        self.assertAlmostEqual(x[0], a)
        self.assertAlmostEqual(v[0], 2*a)

if __name__ == "__main__":
    unittest.main(buffer=False)
