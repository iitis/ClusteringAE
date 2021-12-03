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

import pickle
from typing import Tuple, Dict
from collections import namedtuple
from pandas import DataFrame


def analyze_ate_results(
    fpath: str = '/home/user/ClusteringAE/results/results.pkl',
):
    """
    Analyze ATE results.

    :param fpath: path to results file.
    """
    Params = namedtuple('ColumnsFilter', ['column', 'rows'])
    with open(fpath, 'rb') as f:
        results = pickle.load(f)

    print('MAIN EXP')
    print('\tSAMSON')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Samson',)),
            Params('run', (0,))
        )
    )
    print('\tJASPER')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Jasper',)),
            Params('run', (0,))
        )
    )
    print('\n')

    print('WEIGHTS INITIALIZATIONS')
    print('\tSAMSON')
    print('\tKaiming He normal')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Samson',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Kaiming_He_normal',))
        )
    )
    print('\tKaiming He uniform')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Samson',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Kaiming_He_uniform',))
        )
    )
    print('\tXavier Glorot normal')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Samson',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Xavier_Glorot_normal',))
        )
    )
    print('\tXavier Glorot uniform')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Samson',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Xavier_Glorot_uniform',))
        )
    )
    print('\tJASPER')
    print('\tKaiming He normal')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Jasper',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Kaiming_He_normal',))
        )
    )
    print('\tKaiming He uniform')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Jasper',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Kaiming_He_uniform',))
        )
    )
    print('\tXavier Glorot normal')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Jasper',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Xavier_Glorot_normal',))
        )
    )
    print('\tXavier Glorot uniform')
    get_results(
        data=results,
        params=(
            Params('architecture', ('basic',)),
            Params('loss_type', ('MSE',)),
            Params('dataset', ('Jasper',)),
            Params('run', (0,)),
            Params('weight_initialization', ('Xavier_Glorot_uniform',))
        )
    )
    print('\n')


def get_results(
    data: DataFrame,
    params: Tuple,
    metrics: Tuple[str] = (
        'reconstruction_error_RMSE',
        'endmembers_error',
        'abundances_error',
    )
) -> Dict[str, Dict[str, float]]:
    """
    Get results for exp described by params.

    :param data: pandas DataFrame containing all exps results.
    :param params: params describing exp to get results for.
    :param metrics: metrics to print.
    :return: dictionary of filtered results.
    """
    results = {}
    filtered_data = data.copy()
    for p in params:
        filtered_data = filter_rows(
            data=filtered_data,
            filter_col=p.column,
            rows=p.rows
        )
    for metric in metrics:
        metric_value = filtered_data[metric]
        mean_ = metric_value.mean()
        std_ = metric_value.std(ddof=0)
        print(f'\t{metric}\t${mean_:.3f}\pm{std_:.2f}$')
        results[metric] = {}
        results[metric]['mean'] = mean_
        results[metric]['std'] = std_
    print('\n')
    return results


def filter_rows(
    data: DataFrame,
    filter_col: str,
    rows: Tuple
) -> DataFrame:
    """
    Filter rows in pandas DataFrame.

    :param data: pandas DataFrame to filter.
    :param filter_col: name of column to filter rows by.
    :param rows: values of rows from filter_col to include.
    :return: filtered pandas DataFrame.
    """
    return data.loc[data[filter_col].isin(rows)]


if __name__ == "__main__":
    analyze_ate_results()
