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

from grids.models import BasicEncoder


class TestBasicEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = BasicEncoder(
            n_bands=123,
            n_endmembers=7,
            l_n=5,
            n_classes=9
        )
    def test___init__(self):
        self.assertEqual(self.enc.n_bands, 123)
        self.assertEqual(self.enc.n_endmembers, 7)
        params = self.enc.parameters()
        l1 = next(params)
        self.assertEqual(l1.shape, (5*7, 123))
        b1 = next(params)
        self.assertEqual(b1.shape, (5*7,))
        l2 = next(params)
        self.assertEqual(l2.shape, (7, 5*7))
        b2 = next(params)
        self.assertEqual(b2.shape, (7,))
        o = next(params)
        self.assertEqual(o.shape, (9, 7))
        b3 = next(params)
        self.assertEqual(b3.shape, (9,))
    def test_forward(self):
        out = self.enc(torch.FloatTensor([i for i in range(123)]))
        self.assertEqual(out.shape, (9,))


if __name__ == "__main__":
    unittest.main()
