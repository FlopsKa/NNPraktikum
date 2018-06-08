import unittest
import numpy as np
from util.activation_functions import Activation

class TestActivationFunctions(unittest.TestCase):

    def test_softmax_activation(self):
        activations = np.array([0, 1, 2])
        result = Activation.softmax(activations)
        print(result)
        self.assertEqual(np.sum(np.array([0.09003057, 0.24472847, 0.66524096])), 1.0)

if __name__ == '__main__':
    unittest.main()
    