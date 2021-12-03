"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import torch.nn.functional as F

from torch import nn
from ray import tune

from util_nn import sum_to_one_constraint

# ------------------------CONST ------------------------------------------------

#class name of the autoencoder class (you can leave it as it is)
AA_CLASS_NAME = 'Autoencoder'

# ------------------------AA CLASS------------------------------------------------

#autoencoder class
class Autoencoder(nn.Module):

    def __init__(self, n_bands, n_endmembers, l_n=10):
        """
        Simple autoencoder (with relu)

        Parameters
        bands: no. bands
        endmembers: no. endmembers
        l_n: no. neurons in the 2nd layer is (l_n*endmembers)
        """
        super(Autoencoder, self).__init__()
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.linear1 = nn.Linear(n_bands, l_n * self.n_endmembers)
        self.linear2 = nn.Linear(l_n * self.n_endmembers, self.n_endmembers)
        self.linear5 = nn.Linear(self.n_endmembers, n_bands, bias=False)
        self.params_grid = {
            "batch_size": tune.choice([5, 20, 25, 30, 100, 250]),
            "learning_rate": tune.loguniform(1e-4, 1e-1, 10),
            "weight_decay": tune.choice([1e-5, 1e-4, 1e-3]),
            "l_n": tune.choice([5, 10, 15, 30, 50]),
        } if tune is not None else None
        # possible options for activation function: 
        # {"function": 'sigmoid', 'tanh', 'relu', 'leaky_relu'
        #  "param": None or float (in the case of leaky_relu)}
        self.activation_function = {
            'function': 'relu',
            'param': None
        }

    def forward(self, x):
        # encoder
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        abundances = sum_to_one_constraint(x, self.n_endmembers)
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
