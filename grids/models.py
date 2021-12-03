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

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear


class BasicEncoder(Module):

    def __init__(
        self,
        n_bands: int,
        n_endmembers: int,
        l_n: int = 10,
        n_classes: int = 9
    ):
        """
        Encoder part of the Basic autoencoder with addition of classification layer.

        :param n_bands: no. bands
        :param n_endmembers: no. endmembers
        :param l_n: no. neurons in the 2nd layer is (l_n*endmembers)
        :param n_classes: no. classes
        """
        super(BasicEncoder, self).__init__()
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.linear1 = Linear(n_bands, l_n * self.n_endmembers)
        self.linear2 = Linear(l_n * self.n_endmembers, self.n_endmembers)
        self.output = Linear(self.n_endmembers, n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process data through the network

        :param x: data to process
        :return: processed data
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)
