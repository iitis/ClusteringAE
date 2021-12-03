"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from torch.utils.data import DataLoader

from ate_tests.test_params import get_params
from ate.ate_core import _update_dataset_info
from ate.ate_autoencoders import (get_autoencoder, get_trained_network, _aa_factory)
from ate.ate_data import get_dataset
from architectures import basic

class Test(unittest.TestCase):
    def setUp(self):
        self.autoencoder_name = "basic"
        self.dataset_name = 'Custom'
        params_global, _ = get_params()
        self.dataset = get_dataset(self.dataset_name,
                                   path=params_global['path_data'],
                                   normalisation='max')

    def test__aa_factory(self):
        params_global, params_aa = get_params()
        _update_dataset_info(params_global, self.dataset)
        aut = _aa_factory(basic.Autoencoder, params_aa, params_global)
        self.assertIsInstance(aut, basic.Autoencoder)
        self.assertEqual(aut.n_bands, params_global["n_bands"])
        self.assertEqual(aut.n_endmembers, params_global["n_endmembers"])

    def test_get_autoencoder(self):
        params_global, params_aa = get_params()
        _update_dataset_info(params_global, self.dataset)
        aut = get_autoencoder(self.autoencoder_name, params_aa, params_global)
        self.assertIsInstance(aut, basic.Autoencoder)
        self.assertEqual(aut.params_grid, aut.get_params_grid())
        self.assertEqual(aut.n_bands, params_global["n_bands"])
        self.assertEqual(aut.n_endmembers, params_global["n_endmembers"])

    def test_get_trained_network(self):
        params_global, params_aa = get_params()
        _update_dataset_info(params_global, self.dataset)
        data_loader = DataLoader(self.dataset,
                                 batch_size=params_aa["batch_size"],
                                 shuffle=True)
        net = get_trained_network(autoencoder_name=self.autoencoder_name,
                                  data_loader=data_loader,
                                  params_aa=params_aa,
                                  params_global=params_global,
                                  n_runs=2)
        self.assertIsInstance(net, basic.Autoencoder)
        self.assertEqual(net.n_bands, params_global["n_bands"])
        self.assertEqual(net.n_endmembers, params_global["n_endmembers"])


if __name__ == "__main__":
    print ("Run tests from ../run_ate_tests.py")
