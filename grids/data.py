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

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from torch.utils.data import Dataset

from ATE.ate.ate_data import spectra_normalisation


class Dataset(Dataset):
    def __init__(
        self,
        dpath: str = 'ATE/data/Samson.npz',
        data_hw: Tuple[int] = (95, 95),
        grid_shape: Tuple[int] = (3, 3),
        normalisation: str = None
    ):
        """
        Initialize Dataset object.

        :param dpath: path to data
        :param data_hw: height and width of the original image
        :param grid_shape: shape of the grid to generate as a GT
        :param normalisation: data normalisation [minmax/max/max1]
        """
        self.img_height, self.img_width = data_hw
        self.grid_shape = grid_shape
        self.normalisation = normalisation
        self.X = np.load(dpath, allow_pickle=True)['data']
        self.n_bands = self.X.shape[-1]
        img = self.X.reshape(self.img_height, self.img_width, self.n_bands)
        self.y = self._create_grid(img.shape, grid_shape)
        self.y = self.y.reshape(-1)
        if normalisation is not None:
            self.X, _ = spectra_normalisation(self.X,
                                              normalisation)
        del img

    def __getitem__(self, index: int) -> Tuple[np.array, np.float64]:
        """
        Return sample from dataset.

        :param index: index of the sample to return
        :return: sample data and its GT
        """
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        """
        Length of the dataset, i.e. no. samples.

        :return: length of the dataset
        """
        return len(self.X)

    def _create_grid(
        self,
        data_shape: Tuple[int],
        grid_shape: Tuple[int]
    ) -> np.array:
        """
        Create grid ground truth.

        :param data_shape: shape of the original image
        :param grid_shape: shape of the grid to generate as a GT
        :return: grid ground truth
        """
        assert len(data_shape) >= 2
        if len(data_shape) > 2:
            data_shape = data_shape[:2]
        gt = np.zeros(data_shape)
        class_height = int(np.ceil(data_shape[0] / grid_shape[0]))
        class_width = int(np.ceil(data_shape[1] / grid_shape[1]))
        k = 0
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                gt[i*class_height:(i+1)*class_height,
                j*class_width:(j+1)*class_width] = k
                k += 1
        return gt


def data_demo():
    """
    Demo of the Dataset class.
    """
    data = Dataset()
    plt.imshow(data.X.reshape(
        data.img_height, data.img_width, data.n_bands)[:, :, 100])
    plt.show()
    plt.imshow(data.y.reshape(data.img_height, data.img_width))
    plt.show()
    print(data[0])


if __name__ == "__main__":
    data_demo()
