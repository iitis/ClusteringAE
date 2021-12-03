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

import math
import numpy as np
from typing import Tuple, Dict, List

from grids.utils import get_exp_data


def get_good_bad_models_results(
    dname: str = 'Samson',
    grid_size: int = 3,
    metrics: Tuple[str] = (
        'reconstruction_error_RMSE_multiplication',
        'endmembers_error',
        'abundances_error_multiplication',
    )
) -> Tuple[Dict[str, List[float]]]:
    """
    Script for getting results for good & bad models from MLFlow.

    :param dname: name of the dataset.
    :param grid_size: size of the grid.
    :param metrics: metrics to get.

    :return: dicts containing metrics for good and bad models, respectively.
    """
    print(dname, f'{grid_size}x{grid_size}')
    if dname == 'Samson':
        good_models = {
            'Xavier_Glorot_normal': [24, 5, 13, 15, 47],
            'Xavier_Glorot_uniform': [16, 24, 4, 1, 23],
            'Kaiming_He_normal': [10, 5, 49, 47, 25],
            'Kaiming_He_uniform': [0, 40, 9, 24, 30]
        }
        bad_models = {
            'Xavier_Glorot_normal': [35, 0, 12, 44, 31],
            'Xavier_Glorot_uniform': [5, 13, 25, 18, 31],
            'Kaiming_He_normal': [35, 0, 12, 44, 31],
            'Kaiming_He_uniform': [28, 5, 25, 18, 26]
        }
    if dname == 'Jasper':
        good_models = {
            'Xavier_Glorot_normal': [16, 12, 19, 10, 17],
            'Xavier_Glorot_uniform': [33, 6, 31, 30, 21],
            'Kaiming_He_normal': [19, 47, 29, 1, 40],
            'Kaiming_He_uniform': [41, 30, 6, 10, 37]
        }
        bad_models = {
            'Xavier_Glorot_normal': [30, 20, 0, 39, 42],
            'Xavier_Glorot_uniform': [32, 1, 24, 45, 14],
            'Kaiming_He_normal': [20, 30, 0, 39, 42],
            'Kaiming_He_uniform': [44, 16, 20, 46, 31]
        }
    good_metrics = {}
    bad_metrics = {}
    for metric in metrics:
        good_metrics[metric] = []
        bad_metrics[metric] = []
    exp_data = get_exp_data(
        exp_name=f'{dname}_({grid_size},{grid_size})'
    )
    for exp in exp_data:
        if int(exp['params']['model_no']) in good_models[exp['params']['weights_init']]:
            for metric in metrics:
                good_metrics[metric].append(exp['metrics'][metric])
        if int(exp['params']['model_no']) in bad_models[exp['params']['weights_init']]:
            for metric in metrics:
                bad_metrics[metric].append(exp['metrics'][metric])
    print('GOOD MODELS')
    for metric in metrics:
        mean_ = np.mean(good_metrics[metric])
        std_ = np.std(good_metrics[metric])
        print(f'\t{metric:50s}\t${mean_:.3f}\pm{std_:.2f}$')
    print('BAD MODELS')
    for metric in metrics:
        mean_ = np.mean(bad_metrics[metric])
        std_ = np.std(bad_metrics[metric])
        print(f'\t{metric:50s}\t${mean_:.3f}\pm{std_:.2f}$')
    print('\n')
    return good_metrics, bad_metrics


def test_get_good_bad_models_results():
    good_metrics, _ = get_good_bad_models_results(
        dname='Jasper', grid_size=7, metrics=('endmembers_error',)
    )
    assert math.isclose(
        np.mean(good_metrics['endmembers_error']),
        0.963440952385879,
        rel_tol=1e-10
    )

    _, bad_metrics = get_good_bad_models_results(
        dname='Samson', grid_size=3,
        metrics=('reconstruction_error_RMSE_multiplication',)
    )
    assert math.isclose(
        np.std(bad_metrics['reconstruction_error_RMSE_multiplication']),
        0.051156079590009,
        rel_tol=1e-10
    )


if __name__ == "__main__":
    for dname in ('Samson', 'Jasper'):
        for grid_size in (3, 5, 7):
            get_good_bad_models_results(
                dname=dname,
                grid_size=grid_size
            )
    test_get_good_bad_models_results()
