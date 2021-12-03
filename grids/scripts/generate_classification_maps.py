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
import mlflow
import uuid
import functools
import torch
import matplotlib.pyplot as plt
from typing import Tuple
from pathlib import Path

from grids.data import Dataset
from grids.models import BasicEncoder
from grids.utils import get_exp_data
from cfg.params_global import params_global
from ATE.ate.ate_utils import load_model


def gen_class_maps(
    exp_name: str,
    tracking_uri: str = 'file:/home/user/ClusteringAE/mlruns',
    rpath: Path = Path('results/'),
    metrics: Tuple[str] = (
        'reconstruction_error_RMSE_multiplication',
        'endmembers_error',
        'abundances_error_multiplication'
    ),
):
    """
    Generate classification maps.

    :param exp_name: name of the experiment
    :param tracking_uri: path to MLFlow data
    :param rpath: path to save visualisations
    :param metrics: metrics to report
    """
    # Load exp data
    exps = get_exp_data(
        exp_name = exp_name,
        tracking_uri = tracking_uri
    )
    for exp_data in exps:
        # Create dirs
        rpath_temp = rpath / uuid.uuid4().hex
        rpath_temp.mkdir(parents=True, exist_ok=False)
        # MLFlow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name+'_class_maps')
        mlflow.start_run(run_name=exp_data['run_id'])
        mlflow.log_params(exp_data['params'])
        mlflow.log_metrics(exp_data['metrics'])
        # Params
        dpath = exp_data['params']['dpath']
        dname = exp_data['params']['dname']
        grid_shape = eval(exp_data['params']['grid_shape'])
        l_n = int(exp_data['params']['l_n'])
        if dname == 'Samson':
            cube_shape = (95, 95, 156)
            n_endmembers = 3
        elif dname == 'Jasper':
            cube_shape = (100, 100, 198)
            n_endmembers = 4
        # Load data
        normalisation = 'max' if 'normalisation' not in params_global \
            else params_global['normalisation']
        data = Dataset(
            dpath=dpath + dname + '.npz',
            data_hw=cube_shape[:2],
            grid_shape=grid_shape,
            normalisation=normalisation
        )
        # Load model
        model = BasicEncoder(
            n_bands=data.n_bands,
            n_endmembers=n_endmembers,
            l_n=l_n,
            n_classes=functools.reduce(lambda x,y: x*y, grid_shape)
        )
        weights_path = exp_data['artifact_uri'] / 'pretrain_models' / 'net_fin.pt'
        model = load_model(weights_path, model, strict=True)
        # Evaluate data
        pred = model(torch.from_numpy(data.X).to(
                dtype=torch.float
            ))
        pred = torch.argmax(pred, dim=-1)
        pred = pred.reshape(cube_shape[:2])
        # GT for visualisation
        gt = data.y.reshape(cube_shape[:2])
        # Create visualisations
        plt.figure()
        plt.imshow(pred, cmap='tab20', interpolation='nearest')
        plt.savefig(rpath_temp / 'pred.png')
        plt.close()

        plt.figure()
        plt.imshow(gt, cmap='tab20', interpolation='nearest')
        plt.savefig(rpath_temp / 'gt.png')
        plt.close()

        fig, axs = plt.subplots(1, 2)
        fig.suptitle(exp_name+'-'+exp_data['run_id'])
        axs[0].imshow(pred, cmap='tab20', interpolation='nearest')
        axs[1].imshow(gt, cmap='tab20', interpolation='nearest')
        desc = ''
        for m in metrics:
            desc += f'{m}: ' + str(exp_data['metrics'][m]) + '\n'
        fig.text(.5, .05, desc, ha='center')
        plt.savefig(rpath_temp / f'{exp_data["run_id"]}.png')
        plt.close()
        # MLFlow
        mlflow.log_artifacts(rpath_temp)
        shutil.rmtree(rpath_temp)
        mlflow.end_run()


def download_class_maps_from_mlflow(
    exp_name: str,
    tracking_uri: str = 'file:/home/user/ClusteringAE/mlruns',
    rpath: Path = Path('results/')
):
    """
    Download classification maps from MLFlow.

    :param exp_name: name of the experiment
    :param tracking_uri: path to MLFlow dir
    :param rpath: path to download maps into
    """
    (rpath / exp_name).mkdir(parents=True, exist_ok=False)
    exps = get_exp_data(
        exp_name = exp_name,
        tracking_uri = tracking_uri
    )
    for exp_data in exps:
        artifact_path = Path(exp_data['artifact_uri'])
        run_id = exp_data['run_id']
        model_no = exp_data['params']['model_no']
        weights_init = exp_data['params']['weights_init']
        shutil.copyfile(
            artifact_path / 'pred.png',
            rpath / exp_name / f'{weights_init}_{model_no}_{run_id}.png'
        )


if __name__ == "__main__":
    for d in ('Samson', 'Jasper'):
        for g in ('(3,3)', '(5,5)', '(7,7)'):
            # gen_class_maps(
            #     exp_name = f'{d}_{g}'
            # )
            download_class_maps_from_mlflow(
                exp_name = f'{d}_{g}_class_maps'
            )
