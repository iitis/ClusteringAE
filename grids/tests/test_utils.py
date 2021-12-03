"""
Copyright 2021 Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
Authors:
- Bartosz Grabowski (ITAI PAS, ORCID ID: 0000−0002−2364−6547)
- Przemysław Głomb (ITAI PAS, ORCID ID: 0000−0002−0215−4674),
- Kamil Książek (ITAI PAS, ORCID ID: 0000−0002−0201−6220),
- Krisztián Buza (Sapientia Hungarian University of Transylvania, ORCID ID: 0000-0002-7111-6452)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
---
Autoencoders pretraining using clustering v.1.0
Related to the work:
Improving Autoencoders Performance for Hyperspectral Unmixing using Clustering
Source code for the review process of the 14th Asian Conference on Intelligent
Information and Database Systems (ACIIDS 2022)
"""

import unittest
import torch

from ATE.ate.ate_core import _update_dataset_info
from ATE.ate.ate_data import get_dataset
from ATE.ate.ate_utils import load_model
from ATE.ate.ate_autoencoders import get_autoencoder
from grids.utils import tuple_converter, get_basic_encoder, get_ov_acc
from cfg.params_global import params_global
from cfg.default_params_aa import default_params_aa


DPATH = 'ATE/data/'
DNAME = 'Samson'
MPATH = ('/home/user/ClusteringAE/models')


class TestUtils(unittest.TestCase):
    def test_tuple_converter(self):
        out = tuple_converter('(2, 4)')
        self.assertEqual(out, (2, 4))
        out = tuple_converter('4, 5, 10')
        self.assertEqual(out, (4, 5, 10))

    def test_get_basic_encoder(self):
        params_global['path_data'] = DPATH
        normalisation = 'max' if 'normalisation' not in params_global \
            else params_global['normalisation']
        dataset = get_dataset(name=DNAME,
                            path=params_global['path_data'],
                            normalisation=normalisation)
        _update_dataset_info(params_global, dataset)
        model = get_autoencoder('basic',
                                default_params_aa,
                                params_global)
        model = load_model(MPATH, model)
        encoder = get_basic_encoder(
            MPATH,
            n_bands=dataset.cube.shape[-1],
            n_endmembers=dataset.n_endmembers,
            l_n=10,
            n_classes=9
            )
        self.assertEqual(model.linear1.weight.ne(encoder.linear1.weight).sum(), 0)
        self.assertEqual(model.linear2.weight.ne(encoder.linear2.weight).sum(), 0)
        self.assertEqual(model.linear1.bias.ne(encoder.linear1.bias).sum(), 0)
        self.assertEqual(model.linear2.bias.ne(encoder.linear2.bias).sum(), 0)

    def test_get_ov_acc(self):
        y_pred = torch.FloatTensor([
            [0.1, 0.9, 1.2],
            [1, 0, 0.4],
            [0.2, 0.2, 0.3]
        ])
        y_true = torch.Tensor([2, 0, 0])
        self.assertAlmostEqual(get_ov_acc(y_pred, y_true), 2/3)


if __name__ == "__main__":
    unittest.main()
