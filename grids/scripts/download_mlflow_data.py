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
from pathlib import Path
from collections import defaultdict

from grids.utils import get_exp_data
from cfg.params_global import params_global
from ATE.ate.ate_data import get_dataset
from ATE.ate.ate_evaluation import (
    mean_RMSE_error, SAD_distance, repair_order_of_pixels
)


def download_mlflow_data(
    dname: str = 'Samson',
    grid_size: int = 3,
    rpath: Path = Path('results/exp_data')
):
    """
    Script for downloading data from MLFlow.

    :param dname: name of the dataset.
    :param grid_size: size of the grid.
    :param rpath: path to save results.
    """
    exp_data = get_exp_data(
        exp_name=f'{dname}_({grid_size},{grid_size})'
    )
    results_total = defaultdict(dict)
    for run in exp_data:
        results = {}
        results['run_id'] = run['run_id']
        results['dname'] = run['params']['dname']
        results['grid_shape'] = run['params']['grid_shape']
        results['model_no'] = run['params']['model_no']
        results['weights_init'] = run['params']['weights_init']
        run_artifacts = np.load(
            run['artifact_uri'] / f'{dname}_({grid_size},{grid_size})_{run["params"]["run_name"]}_basic_{dname}_results.npz'
        )
        abundances = run_artifacts['abundances']
        endmembers = run_artifacts['endmembers']
        results['A_out'] = abundances
        results['E_out'] = endmembers
        results['X_out'] = np.matmul(abundances, endmembers)
        results_total[results['weights_init']][results['model_no']] = results
    np.savez(
        file=rpath / f'{dname}_({grid_size},{grid_size})',
        **results_total
    )


def save_original_data(
    dname: str = 'Samson',
    dpath: str = 'ATE/data/',
    rpath: str = 'results/exp_data/'
):
    """
    Script for saving original datasets.

    :param dname: name of the dataset.
    :param dpath: path to dataset.
    """
    dataset = get_dataset(name=dname,
                          path=dpath,
                          normalisation=params_global['normalisation'])
    _, X = dataset.get_original()
    if dname == 'Samson':
        cube_shape = (95, 95, 156)
    elif dname == 'Jasper':
        cube_shape = (100, 100, 198)
    A = dataset.get_abundances_gt()
    A = repair_order_of_pixels(
        A, cube_shape, dname
    )
    np.savez(
        rpath + f'{dname}_orig.npz',
        X_in=X,
        A_ref=A,
        E_ref=dataset.get_endmembers_gt()
    )


def test_download_mlflow_data(
    rpath: Path
):
    """
    Test for script downloading MLFlow data.

    :param rpath: path to downloaded data.
    """
    S = np.load(rpath / 'Samson_orig.npz', allow_pickle=True)
    J = np.load(rpath / 'Jasper_orig.npz', allow_pickle=True)

    S_3_XGN_10 = np.load(rpath / 'Samson_(3,3).npz', allow_pickle=True) \
                    ['Xavier_Glorot_normal'].item()['10']['X_out']
    v = mean_RMSE_error(S['X_in'], S_3_XGN_10)
    v_true = 0.006667191
    print(v, '=', v_true)
    assert math.isclose(v, v_true, rel_tol=1e-7)

    S_7_KHN_12 = np.load(rpath / 'Samson_(7,7).npz', allow_pickle=True) \
                    ['Kaiming_He_normal'].item()['12']['A_out']
    v = mean_RMSE_error(S_7_KHN_12, S['A_ref'])
    v_true = 0.362638161374508
    print(v, '=', v_true)
    assert math.isclose(v, v_true, rel_tol=1e-7)

    J_5_XGU_30 = np.load(rpath / 'Jasper_(5,5).npz', allow_pickle=True) \
                    ['Xavier_Glorot_uniform'].item()['30']['E_out']
    v = SAD_distance(J_5_XGU_30, J['E_ref'])
    v_true = 0.824496815307149
    print(v, '=', v_true)
    assert math.isclose(v, v_true, rel_tol=1e-7)


if __name__ == "__main__":
    rpath = Path('results/exp_data/')
    # rpath.mkdir(parents=True, exist_ok=False)
    # for dname in ('Samson', 'Jasper'):
    #     save_original_data(
    #         dname=dname,
    #         rpath=str(rpath) + '/'
    #     )
    #     for grid_size in (3, 5, 7):
    #         download_mlflow_data(
    #             dname=dname,
    #             grid_size=grid_size,
    #             rpath=rpath
    #         )
    test_download_mlflow_data(rpath=rpath)
