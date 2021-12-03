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

import shutil
import uuid
import mlflow
from pathlib import Path
from clize import run

from grids.exp_base import exp as exp_base
from grids.exp_pretrain import exp as exp_pretrain
from grids.utils import tuple_converter
from ATE.ate.ate_utils import set_seed, save_model


def exp(
    *,
    exp_name: str,
    run_name: str,
    dpath: str = 'ATE/data/',
    dname: str = 'Samson',
    mpath: str = '/home/user/ClusteringAE/models/',
    model_no: int = 0,
    rpath: str = 'results/',
    grid_shape: tuple_converter = (3, 3),
    weights_init: str = 'Kaiming_He_uniform',
    l_n: int = 10,
    n_checkpoints: int = 10,
    seed: int = None,
    use_mlflow: bool = False,
    mlflow_path: str = "/home/user/ClusteringAE/mlruns",
):
    """
    Performs and returns the results of AE unmixing with
    Grids clustering pretraining.

    :param exp_name: name of the experiment
    :param run_name: name of the run
    :param dpath: path to data
    :param dname: name of the dataset
    :param mpath: path to the models
    :param model_no: number of model.
    :param rpath: results path
    :param grid_shape: shape of the grid to generate as a GT
    :param weights_init: initialization of the weights
    :param l_n: no. neurons in the 2nd layer of the encoder; is (l_n*endmembers)
    :param n_checkpoints: no. saved checkpoints of pretrained encoder
    :param seed: random seed
    :param use_mlflow: whether to use MLFlow
    :param mlflow_path: path to MLFlow dir
    """
    print(locals())
    # MLFlow & results dir & paths preparation
    rpath = rpath + uuid.uuid4().hex + '/'
    print('Results path:', rpath)
    Path(rpath).mkdir(parents=True, exist_ok=False)
    if use_mlflow:
        mlflow.set_tracking_uri(mlflow_path)
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(locals())
    if dname == 'Samson':
        exp_id = 4
    elif dname == 'Jasper':
        exp_id = 10
    mpath = mpath + f'exp_{weights_init}_MSE_{exp_id}_verification_basic_{dname}/' \
            + f'model_{model_no}/model_initialization'
    # Set seed
    if seed is not None:
        set_seed(seed)
    # Pretrain model
    model = exp_pretrain(
        dpath=dpath,
        dname=dname,
        mpath=mpath,
        rpath=rpath,
        grid_shape=grid_shape,
        weights_init=weights_init,
        l_n=l_n,
        n_checkpoints=n_checkpoints,
        use_mlflow=use_mlflow,
    )
    save_model(path=rpath + 'pretrained_ae.pt', model=model)
    # Perform unmixing
    evaluation_result = exp_base(
        exp_name=exp_name,
        run_name=run_name,
        dpath=dpath,
        dname=dname,
        mpath=rpath + 'pretrained_ae.pt',
        rpath=rpath,
        weights_init=weights_init
    )
    # MLFlow
    if use_mlflow:
        mlflow.log_metrics(evaluation_result)
        mlflow.log_artifacts(rpath)
        shutil.rmtree(rpath)
        mlflow.end_run()


if __name__ == "__main__":
    run(exp)
