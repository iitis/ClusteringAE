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

import mlflow
import torch
from typing import Tuple, List, Dict
from pathlib import Path
from clize.errors import ArgumentError
from clize import parser

from ATE.ate.ate_utils import load_model
from grids.models import BasicEncoder


@parser.value_converter
def tuple_converter(arg: str) -> Tuple:
    out = eval(arg)
    if type(out) is tuple:
        return out
    else:
        raise ArgumentError(f'{arg} must be a str convertable to tuple')


def get_basic_encoder(
    mpath: str,
    n_bands: int,
    n_endmembers: int,
    l_n: int,
    n_classes: int
    ) -> BasicEncoder:
    """
    Returns encoder part of the Basic autoencoder

    :param mpath: path to the Basic autoencoder object state dict
    :param n_bands: no. bands
    :param n_endmembers: no. endmembers
    :param l_n: no. neurons in the 2nd layer is (l_n*endmembers)
    :param n_classes: no. classes
    :return: encoder part of the Basic autoencoder
    """
    enc = BasicEncoder(
        n_bands=n_bands,
        n_endmembers=n_endmembers,
        l_n=l_n,
        n_classes=n_classes)
    return load_model(mpath, enc, strict=False)


def get_ov_acc(y_pred: torch.FloatTensor, y_true: torch.Tensor) -> float:
    """
    Get overall accuracy

    :param y_pred: model predictions
    :param y_true: data labels
    :return: overall accuracy score
    """
    acc = torch.sum(y_true == torch.argmax(y_pred, dim=1))/float(len(y_true))
    return acc.item()


def get_exp_data(
    exp_name: str,
    tracking_uri: str = 'file:mlruns'
) -> List[Dict]:
    """
    Get info about experiment runs.

    :param exp_name: name of the experiment
    :param tracking_uri: path to MLFlow dir

    :return: exp info, i.e. run id, path to artifacts, params dict, metrics
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(exp_name)
    run_list = mlflow.list_run_infos(experiment.experiment_id)
    data = []
    for run_info in run_list:
        run = mlflow.get_run(run_info.run_id)
        data.append({
            'run_id': run_info.run_id,
            'artifact_uri': Path(run_info.artifact_uri.replace('file:', '')),
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
        })
    return data
