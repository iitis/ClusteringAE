"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import torch

from torch import nn
from ray import tune

from util_nn import (sum_to_one_constraint, DynamicalSoftThresholding)

# ------------------------CONST ------------------------------------------------

#class name of the autoencoder class (you can leave it as it is)
AA_CLASS_NAME = 'Autoencoder'

# ------------------------AA CLASS------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, n_bands, n_endmembers):
        """
        Simple autoencoder (with relu)

        Parameters
        n_bands: no. bands
        n_endmembers: no. endmembers
        """
        super(Autoencoder, self).__init__()
        self.bands = n_bands
        self.endmembers = n_endmembers
        self.linear1 = nn.Linear(n_bands, 9 * self.endmembers)
        self.linear2 = nn.Linear(9 * self.endmembers, 6 * self.endmembers)
        self.linear3 = nn.Linear(6 * self.endmembers, 3 * self.endmembers)
        self.linear4 = nn.Linear(3 * self.endmembers, self.endmembers)
        self.bn1 = nn.BatchNorm1d(num_features=self.endmembers)
        self.soft_thresholding = DynamicalSoftThresholding([self.endmembers])
        self.linear5 = nn.Linear(self.endmembers, n_bands, bias=False)
        self.params_grid = {
            "batch_size": tune.choice([5, 10, 20, 25, 30, 100, 250]),
            "learning_rate": tune.uniform(1e-4, 1e-1),
            "weight_decay": tune.choice([1e-5, 1e-4, 1e-3, 0]),
        } if tune is not None else None
        # possible options for activation function: 
        # {"function": 'sigmoid', 'tanh', 'relu', 'leaky_relu'
        #  "param": None or float (in the case of leaky_relu)}
        self.activation_function = {
            'function': 'relu',
            'param': None
        }

    def get_endmembers(self):
        return self.linear5.weight.cpu().data.numpy()

    def forward(self, x):
        # encoder
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = self.soft_thresholding(self.bn1(x))
        abundances = sum_to_one_constraint(x, self.endmembers)
        # decoder
        x = self.linear5(abundances)
        return (abundances, x)

    def get_params_grid(self):
        """
        Returns parameters designed for this architecture for Grid Search.
        """
        return self.params_grid

    def get_activation_function(self):
        """
        Returns name of the activation function
        """
        return self.activation_function

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass
