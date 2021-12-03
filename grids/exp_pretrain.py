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

import functools
import torch
from pathlib import Path
from clize import run
from torch.utils.data import DataLoader

from grids.data import Dataset
from grids.train import train
from grids.utils import get_basic_encoder, tuple_converter
from cfg.params_global import params_global
from cfg.default_params_aa import default_params_aa
from ATE.ate.ate_core import _update_dataset_info
from ATE.ate.ate_data import get_dataset
from ATE.ate.ate_utils import load_model, get_device
from ATE.ate.ate_autoencoders import get_autoencoder
from ATE.architectures.basic import Autoencoder as BasicAutoencoder


def exp(
    *,
    dpath: str = 'ATE/data/',
    dname: str = 'Samson',
    mpath: str = 'models',
    rpath: str = 'results/',
    grid_shape: tuple_converter = (3, 3),
    weights_init: str = 'Kaiming_He_uniform',
    l_n: int = 10,
    n_checkpoints: int = 10,
    use_mlflow: bool = False
) -> BasicAutoencoder:
    """
    Performs and returns the results of AE unmixing with pretraining.

    :param dpath: path to data
    :param dname: name of the dataset
    :param mpath: path to the model
    :param rpath: results path
    :param grid_shape: shape of the grid to generate as a GT
    :param weights_init: initialization of the weights
    :param l_n: no. neurons in the 2nd layer of the encoder; is (l_n*endmembers)
    :param n_checkpoints: no. saved checkpoints of pretrained encoder
    :param use_mlflow: whether to use MLFlow
    :return: Autoencoder model with pretrained encoder
    """
    # Config
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
    rpath_pretrain = rpath + 'pretrain_models/'
    Path(rpath_pretrain).mkdir(parents=True, exist_ok=False)
    # Load data
    normalisation = 'max' if 'normalisation' not in params_global \
        else params_global['normalisation']
    data = get_dataset(name=params_global['dataset_name'],
                       path=params_global['path_data'],
                       normalisation=normalisation)
    _update_dataset_info(params_global, data)
    # Load pretraining data
    data_pretrain = Dataset(
        dpath=dpath + params_global['dataset_name'] + '.npz',
        data_hw=params_global['cube_shape'][:2],
        grid_shape=grid_shape,
        normalisation=normalisation
    )
    dataloader_pretrain = DataLoader(data_pretrain,
                                     batch_size=default_params_aa["batch_size"],
                                     shuffle=True)
    # Load encoder model
    enc = get_basic_encoder(
        mpath=mpath,
        n_bands=data.cube.shape[-1],
        n_endmembers=data.n_endmembers,
        l_n=l_n,
        n_classes=functools.reduce(lambda x,y: x*y, grid_shape)
        )
    # Train the encoder model
    device = get_device(params_global)
    if device == 'cuda:0' and torch.cuda.device_count() > 1:
        enc = torch.nn.DataParallel(enc)
    enc.to(device)
    if params_global['optim'] == 'adam':
        optimizer = torch.optim.Adam(enc.parameters(),
                                     lr=default_params_aa["learning_rate"],
                                     weight_decay=default_params_aa["weight_decay"])
    elif params_global['optim'] == 'sgd':
        optimizer = torch.optim.SGD(enc.parameters(),
                                    lr=default_params_aa["learning_rate"])
    else:
        raise NotImplementedError
    train(
        trainloader=dataloader_pretrain,
        net=enc,
        optimizer=optimizer,
        epochs=default_params_aa['no_epochs'],
        loss_fn=torch.nn.CrossEntropyLoss(reduction='mean'),
        mpath=rpath_pretrain,
        n_checkpoints=n_checkpoints,
        device=device,
        use_mlflow=use_mlflow
        )
    # Load weights from trained encoder into autoencoder model
    ae = get_autoencoder('basic',
                         default_params_aa,
                         params_global)
    ae = load_model(path=mpath, model=ae, strict=True)
    ae = load_model(path=rpath_pretrain + 'net_fin.pt', model=ae, strict=False)
    return ae


if __name__ == "__main__":
    run(exp)
