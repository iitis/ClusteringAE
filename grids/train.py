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

import torch
from typing import Callable
from mlflow import log_metric
from torch.utils import data
from torch.optim import Adam

from grids.utils import get_ov_acc
from ATE.ate.ate_utils import save_model


def train(trainloader: data.DataLoader, net: torch.nn.Module, optimizer: Adam,
          epochs: int, loss_fn: Callable, mpath: str, n_checkpoints: int,
          device: str = 'cpu', use_mlflow: bool = False):
    """
    Train classification model

    :param trainloader: trainloader object
    :param net: model object
    :param optimizer: optimizer for training
    :param epochs: number of epochs
    :param loss_fn: loss function; must be of the form loss_fn(pred, gt)
    :param mpath: path to save model
    :param n_checkpoints: number of checkpoints to test and save during training
    :param device: device to run training on
    :param: use_mlflow: whether to use MLFlow
    """
    for epoch in range(epochs):
        net.train()
        loss_sum = 0.
        for inputs, labels in trainloader:
            inputs = inputs.to(
                device=device,
                dtype=torch.float
            )
            labels = labels.to(
                device=device,
                dtype=torch.long
            )
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
        train_loss = loss_sum/len(trainloader)
        print(f'Epoch {epoch+1}: {round(train_loss, 6)}')
        if use_mlflow:
            log_metric(key='train_loss', value=train_loss, step=epoch+1)
        if (epoch+1) % int(epochs / n_checkpoints) == 0:
            # Test the model
            in_ = torch.from_numpy(trainloader.dataset.X).to(
                device=device,
                dtype=torch.float
            )
            pred = net(in_)
            gt = torch.from_numpy(trainloader.dataset.y).to(
                device=device,
                dtype=torch.long
            )
            ov_acc = get_ov_acc(pred, gt)
            print(f'Overall accuracy (epoch {epoch+1}):', ov_acc)
            if use_mlflow:
                log_metric(key='enc_ov_acc', value=ov_acc, step=epoch+1)
            # Save model
            save_model(
                path=mpath + f'net_{epoch+1}.pt',
                model=net
            )
    save_model(
        path=mpath + 'net_fin.pt',
        model=net
    )
