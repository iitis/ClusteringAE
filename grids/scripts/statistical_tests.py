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

import scipy
import numpy as np
import pandas as pd
from os.path import join
from itertools import product

from ATE.ate.ate_evaluation import (
    mean_RMSE_error,
    SAD_distance
)


def load_ground_truth(image_name):
    """
    Load ground truth image, endmembers and abundances
    for a given image
    """
    location_grid = ''
    data = np.load(f'{location_grid}/{image_name}_orig.npz',
                   allow_pickle=True)
    image = data['X_in']
    abundances = data['A_ref']
    endmembers = data['E_ref']
    return image, abundances, endmembers


def load_grid_single_model(results,
                           init_method,
                           no_of_model):
    """
    Load results from grid experiments, i.e. image, endmembers
    and abundances
    """
    image = results[init_method].item()[str(no_of_model)]['X_out']
    abundances = results[init_method].item()[str(no_of_model)]['A_out']
    endmembers = results[init_method].item()[str(no_of_model)]['E_out']
    return image, abundances, endmembers


def load_ATE_table():
    location_ATE = ''
    path_ATE = (f'{location_ATE}/file.pkl')
    return pd.read_pickle(path_ATE)


def extract_ATE_result(data,
                       experiment_ID,
                       column_name):
    # 'column_name' value should be one of consecutive
    # options: "abundances_error", "endmembers_error"
    # and "reconstruction_error_RMSE"
    selected_data = data[
        (data['ident'].str.contains(experiment_ID)) &
        (data['run'] == 0)
    ][[column_name, 'model', 'weight_initialization']]
    return selected_data.sort_values(
        by=['model', 'weight_initialization']
    )


def create_table_grid(image_name,
                      grid_size):
    """
    Main function for the creation of results for grid experiment
    """
    image_gt, abundances_gt, endmembers_gt = load_ground_truth(image_name)
    result_list = []
    # columns order: 'model', 'weight_initialization',
    # 'reconstruction_error_RMSE', 'abundances_error',
    # 'endmembers_error'
    location_grid = ''
    path_grids = f'{location_grid}/{image_name}_({grid_size},{grid_size}).npz'
    results = np.load(path_grids, allow_pickle=True)
    for model in range(50):
        for init_method in ['Kaiming_He_normal',
                            'Kaiming_He_uniform',
                            'Xavier_Glorot_uniform',
                            'Xavier_Glorot_normal']:
            image, abundances, endmembers = load_grid_single_model(
                results,
                init_method,
                model)
            reconstruction_error = mean_RMSE_error(image_gt, image)
            abundances_error = mean_RMSE_error(abundances_gt, abundances)
            endmembers_error = SAD_distance(endmembers_gt, endmembers)
            result_list.append(
                [model, init_method, reconstruction_error,
                 abundances_error, endmembers_error]
            )
    dataframe = pd.DataFrame(result_list,
                             columns=['model', 'weight_initialization',
                                      'reconstruction_error_RMSE',
                                      'abundances_error',
                                      'endmembers_error'])
    dataframe.to_pickle(f"{image_name}_({grid_size},{grid_size})_table.pkl")
    return dataframe


def load_table_grid(image_name,
                    grid_size):
    return pd.read_pickle(f'{image_name}_({grid_size},{grid_size})_table.pkl')


def test_grid_preparing():
    # 1) Jasper Ridge
    dataframe_Jasper = load_table_grid('Jasper', '5')
    selected_data = dataframe_Jasper[
        (dataframe_Jasper['weight_initialization'] == 'Kaiming_He_normal') &
        (dataframe_Jasper['model'] == 2)
    ]
    RMSE_result = selected_data['reconstruction_error_RMSE'].item()
    abundances_result = selected_data['abundances_error'].item()
    endmembers_result = selected_data['endmembers_error'].item()
    assert np.abs(RMSE_result - 0.158) <= 0.001
    assert np.abs(abundances_result - 0.341) <= 0.001
    assert np.abs(endmembers_result - 1.193) <= 0.001

    # 2) Samson
    dataframe_Samson = load_table_grid('Samson', '3')
    selected_data = dataframe_Samson[
        (dataframe_Samson['weight_initialization'] == 'Xavier_Glorot_uniform') &
        (dataframe_Samson['model'] == 0)
    ]
    RMSE_result = selected_data['reconstruction_error_RMSE'].item()
    abundances_result = selected_data['abundances_error'].item()
    endmembers_result = selected_data['endmembers_error'].item()
    assert np.abs(RMSE_result - 0.007) <= 0.001
    assert np.abs(abundances_result - 0.307) <= 0.001
    assert np.abs(endmembers_result - 0.814) <= 0.001


def test_order_of_cases_for_paired_tests(reconstruction_errors_ATE,
                                         endmembers_errors_ATE,
                                         abundances_errors_ATE,
                                         dataframe_grid):
    """
    Check whether the order of models and samples corresponding
    different weights initializations is the same in two cases:
    ATE and grid
    """
    assert ((reconstruction_errors_ATE['weight_initialization'].values) == (
        dataframe_grid['weight_initialization'].values
    )).all()
    assert ((reconstruction_errors_ATE['model'].values) == (
        dataframe_grid['model'].values
    )).all()
    assert ((endmembers_errors_ATE['weight_initialization'].values) == (
        dataframe_grid['weight_initialization'].values
    )).all()
    assert ((endmembers_errors_ATE['model'].values) == (
        dataframe_grid['model'].values
    )).all()
    assert ((abundances_errors_ATE['weight_initialization'].values) == (
        dataframe_grid['weight_initialization'].values
    )).all()
    assert ((abundances_errors_ATE['model'].values) == (
        dataframe_grid['model'].values
    )).all()


def prepare_statistical_tests():
    """
    Calculate t-test for dependent samples for results
    of two experiments: Samson and Jasper and three
    grid setups
    """
    table = load_ATE_table()
    results_list = []
    # columns order: image, grid, p-value for reconstruction error,
    # p-value for endmembers error, p-value for abundances error
    for image_name in ['Samson', 'Jasper']:
        experiment = 'F004' if image_name == 'Samson' else 'F010'
        reconstruction_errors_ATE = extract_ATE_result(
            table, experiment,
            'reconstruction_error_RMSE')
        endmembers_errors_ATE = extract_ATE_result(
            table, experiment,
            'endmembers_error')
        abundances_errors_ATE = extract_ATE_result(
            table, experiment,
            'abundances_error')
        for grid in ['3', '5', '7']:
            dataframe_grid = load_table_grid(image_name, grid)
            dataframe_grid = dataframe_grid.sort_values(
                by=['model', 'weight_initialization']
            )
            test_order_of_cases_for_paired_tests(reconstruction_errors_ATE,
                                                 endmembers_errors_ATE,
                                                 abundances_errors_ATE,
                                                 dataframe_grid)
            # statistical test returns t-statistic on the position 0.
            # and p-value on the position 1.
            p_value_reconstruction = scipy.stats.wilcoxon(
                reconstruction_errors_ATE['reconstruction_error_RMSE'].values,
                dataframe_grid['reconstruction_error_RMSE'].values
            )[1]
            p_value_abundances = scipy.stats.wilcoxon(
                abundances_errors_ATE['abundances_error'].values,
                dataframe_grid['abundances_error'].values
            )[1]
            p_value_endmembers = scipy.stats.wilcoxon(
                endmembers_errors_ATE['endmembers_error'].values,
                dataframe_grid['endmembers_error'].values
            )[1]
            results_list.append([image_name, grid, p_value_reconstruction,
                                p_value_endmembers, p_value_abundances])
    dataframe = pd.DataFrame(results_list,
                             columns=['image', 'grid_size',
                                      'p_value_reconstruction',
                                      'p_value_endmembers',
                                      'p_value_abundances'])
    return dataframe


if __name__ == "__main__":
    test_grid_preparing()
    image_names = ['Samson', 'Jasper']
    grid_sizes = ['3', '5', '7']
    for image_name, grid_size in product(image_names,
                                         grid_sizes):
        print(f'Calculations for {image_name}, grid size: {grid_size}')
        grid_results = create_table_grid(image_name,
                                         grid_size)
        dataframe_grid = load_table_grid(image_name,
                                         grid_size)
    dataframe_statistical_tests = prepare_statistical_tests()
    dataframe_statistical_tests.to_csv('Wilcoxon_statistical_tests.csv',
                                       index=False)
