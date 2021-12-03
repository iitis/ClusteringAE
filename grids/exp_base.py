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

from clize import run
from pathlib import Path
from typing import Dict

from cfg.params_global import params_global
from cfg.default_params_aa import default_params_aa
from ATE.ate.ate_core import experiment_simple


def exp(
    *,
    exp_name: str,
    run_name: str,
    dpath: str = 'ATE/data/',
    dname: str = 'Samson',
    mpath: str = 'models',
    rpath: str = 'results/',
    weights_init: str = 'Kaiming_He_uniform',
) -> Dict[str, float]:
    """
    Performs and returns the results of AE unmixing.

    :param exp_name: name of the experiment
    :param run_name: name of the run
    :param dpath: path to data
    :param dname: name of the dataset
    :param mpath: path to the model
    :param rpath: results path
    :param weights_init: initialization of the weights
    :return: metrics of AE unmixing performance
    """
    # Dirs
    vpath = rpath + "/vis/"
    Path(vpath).mkdir(parents=True, exist_ok=False)
    # Params
    params_global['path_data'] = dpath
    params_global['dataset_name'] = dname
    if params_global['dataset_name'] == 'Samson':
        params_global['cube_shape'] = (95, 95, 156)
        default_params_aa['batch_size'] = 4
        default_params_aa['learning_rate'] = 0.0001
    elif params_global['dataset_name'] == 'Jasper':
        params_global['cube_shape'] = (100, 100, 198)
        default_params_aa['batch_size'] = 20
        default_params_aa['learning_rate'] = 0.001
    elif dname == 'Custom':  # Used to run demo version of the experiment.
        params_global['cube_shape'] = (100, 100, 198)
        default_params_aa['batch_size'] = 20
        default_params_aa['learning_rate'] = 0.001
    params_global['weights_init'] = weights_init
    params_global['path_results'] = rpath
    params_global['path_visualisations'] = vpath
    # Run experiment
    evaluation_result, _, _, _, _, _ = experiment_simple(
         autoencoder_name=params_global['autoencoder_name'],
         dataset_name=params_global['dataset_name'],
         params_aa=default_params_aa,
         params_global=params_global,
         loss_function=params_global['loss_type'],
         model_path=mpath,
         mode='train',
         experiment_name=exp_name+"_"+run_name,
         n_runs=1
    )
    return evaluation_result


if __name__ == "__main__":
    run(exp)
