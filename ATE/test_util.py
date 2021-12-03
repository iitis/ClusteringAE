"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import torch
import numpy as np
import unittest

from legacy_util_nn import sum_to_one_constraint, \
    DynamicalSoftThresholding, volume_simplex


class Test(unittest.TestCase):
    def test_sum_to_one_constraint(self):
        '''
        Test function for sum to one constraint.
        '''
        x1 = torch.from_numpy(np.array([[2, 1, 2], [0, 0, 0], [1, 0, 3], [1, 1, 1]]))
        gt1 = torch.from_numpy(np.array([[0.4, 0.2, 0.4], [0.33, 0.33, 0.33], [0.25, 0, 0.75], [0.33, 0.33, 0.33]]))
        difference = np.absolute(gt1 - sum_to_one_constraint(x1, 3))
        self.assertLess(torch.max(difference).item(), 0.1)

    def test_dynamical_soft_thresholding(self):
        '''
        Test function for dynamic soft thresholding
        '''
        soft_thresholding = DynamicalSoftThresholding([3], alpha=0.2)
        x = torch.tensor(0.6).expand_as(torch.arange(3)).clone()
        result = soft_thresholding(x)
        y = torch.tensor(0.4).expand_as(torch.arange(3)).clone()
        difference = torch.abs(result - y)
        self.assertLess(torch.max(difference).item(), 0.0001)
    
    def test_volume_simplex(self):
        '''
        Test function for the calculation of simplex volume
        '''
        test_1 = torch.Tensor([
            [-1., -1.],
            [5., -1.],
            [1., 2.]
        ])
        result_1 = volume_simplex(test_1).item()
        self.assertAlmostEqual(9., result_1, 5)

        test_2 = torch.Tensor([
            [1., 1., 1.],
            [-1., -1., 1.],
            [-1., 1., -1.],
            [1., -1., -1.]
        ])
        result_2 = volume_simplex(test_2).item()
        self.assertAlmostEqual(2.666667, result_2, 5)

        test_3 = torch.Tensor([
            [1., 1., 1., -1./np.sqrt(5.)],
            [1., -1., -1., -1./np.sqrt(5)],
            [-1., 1., -1., -1./np.sqrt(5)],
            [-1., -1., 1., -1./np.sqrt(5)],
            [0., 0., 0., 4./np.sqrt(5)]
        ])
        result_3 = volume_simplex(test_3).item()
        self.assertAlmostEqual(1.4907, result_3, 4)


if __name__ == "__main__":
    unittest.main()
