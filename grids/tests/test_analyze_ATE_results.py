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
from collections import namedtuple
from pandas import DataFrame

from grids.scripts.analyze_ATE_results import filter_rows, get_results


class TestAnalyzeATEResults(unittest.TestCase):
    def setUp(self):
        self.data = DataFrame(
            {
                'id': [0, 3, 4, 5],
                'id2': ['a', 'b', 'c', 'd'],
                'Metric 1': [0.5, 0.3, np.nan, 1.2],
                'Metric 2': [1, 10, 2, 5],
            }
        )
    def test_filter_rows(self):
        filtered_csv = filter_rows(
            data=self.data,
            filter_col='Metric 2',
            rows=(2, 10)
        ).reset_index(drop=True)
        self.assertTrue(filtered_csv.equals(DataFrame({
            'id': [3, 4],
            'id2': ['b', 'c'],
            'Metric 1': [0.3, np.nan],
            'Metric 2': [10, 2]
        })))
    def test_get_results(self):
        Params = namedtuple('ColumnsFilter', ['column', 'rows'])
        results = get_results(
            data=self.data,
            params=(
                Params('id', (0, 3, 4)),
                Params('id2', ('b', 'c', 'd'))
            ),
            metrics=(
                'Metric 1',
                'Metric 2'
            )
        )
        self.assertDictEqual(
            results,
            {
                'Metric 1': {
                    'mean': 0.3,
                    'std': 0
                },
                'Metric 2': {
                    'mean': 6,
                    'std': 4
                }
            }
        )


if __name__ == "__main__":
    unittest.main()
