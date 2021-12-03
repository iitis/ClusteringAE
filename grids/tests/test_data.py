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
import numpy as np

from grids.data import Dataset


DPATH = 'ATE/data/Samson.npz'
DATA_HW = (95, 95)
N_BANDS = 156


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(
            dpath=DPATH,
            data_hw=DATA_HW,
            grid_shape=(3, 3),
            normalisation='minmax'
        )
    def test___init__(self):
        self.assertEqual(
            self.dataset.img_height, DATA_HW[0]
        )
        self.assertEqual(
            self.dataset.img_width, DATA_HW[1]
        )
        self.assertEqual(
            self.dataset.grid_shape, (3, 3)
        )
        self.assertEqual(
            self.dataset.normalisation, 'minmax'
        )
        self.assertEqual(
            np.max(self.dataset.X), 0.5
        )
        self.assertEqual(
            np.min(self.dataset.X), -0.5
        )
        self.assertEqual(
            self.dataset.X.shape, (DATA_HW[0]*DATA_HW[1], N_BANDS)
        )
        self.assertEqual(
            self.dataset.n_bands, N_BANDS
        )
        self.assertEqual(
            set(self.dataset.y), set([i for i in range(9)])
        )
        self.assertEqual(
            self.dataset.y.shape, (DATA_HW[0]*DATA_HW[1],)
        )
    def test___getitem__(self):
        self.assertEqual(
            self.dataset[10][0].shape, (N_BANDS,)
        )
        self.assertIs(
            type(self.dataset[20][1]), np.float64
        )
    def test___len__(self):
        self.assertEqual(
            len(self.dataset), DATA_HW[0]*DATA_HW[1]
        )
    def test__create_grid(self):
        self.assertTrue(
            np.all(
                self.dataset._create_grid(
                    data_shape=(4, 4),
                    grid_shape=(2, 2)
                ) ==
                np.array([
                    [0., 0., 1., 1.],
                    [0., 0., 1., 1.],
                    [2., 2., 3., 3.],
                    [2., 2., 3., 3.]
                ])
            )
        )
        self.assertTrue(
            np.all(
                self.dataset._create_grid(
                    data_shape=(2, 3),
                    grid_shape=(2, 2)
                ) ==
                np.array([
                    [0., 0., 1.],
                    [2., 2., 3.]
                ])
            )
        )


if __name__ == "__main__":
    unittest.main()
