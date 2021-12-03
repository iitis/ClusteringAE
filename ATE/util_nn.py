"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

# ------------------------------------------------------------------------

class GaussianDropout(nn.Module):
    """
    Implementation of gaussian droput
    """
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = Variable(torch.Tensor([alpha]), requires_grad=True)
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1
            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()
            return x * epsilon
        else:
            return x

# ------------------------------------------------------------------------

class DynamicalSoftThresholding(nn.Module):
    '''
    Implementation of a function neurons = max(0, neurons - alpha)
    alpha is a vector of learnable parameters
    '''
    def __init__(self, no_units, alpha=None):
        '''
        Arguments:
        no_units - a list with shapes (one- or two-dimensional)
        alpha - initial value of parameters
        '''
        super(DynamicalSoftThresholding, self).__init__()
        # one-dimensional case
        if len(no_units) == 1:
            self.shape = no_units[0]
            self.mode = 'one'
        # two-dimensional case
        elif len(no_units) == 2:
            self.shape_1 = no_units[0]
            self.shape_2 = no_units[1]
            self.mode = 'multi'
        # other cases
        else:
            raise ValueError('The expected second arguments is the list of size 1 or 2.')

        # if a vector of parameters was not initialized it is filled by default values
        if alpha is None:
            if self.mode == 'one':
                self.alpha = nn.Parameter(torch.tensor(0.01).expand_as(torch.arange(self.shape)).clone())
            elif self.mode == 'multi':
                self.alpha = nn.Parameter(torch.zeros([self.shape_1, self.shape_2]).add(0.01))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        self.alpha.requires_grad = True
    
    def forward(self, x):
        # values of the alpha vector are multiplied by -1 and then are added into neurons
        # finally, maximum of 0 and difference between neurons and alphas are selected
        parameters = x.add_(self.alpha.mul(-1))
        x = F.relu(parameters)
        return x

# ------------------------------------------------------------------------

def sum_to_one_constraint(data, endmembers):
    '''
    Function for a constrain:
    All elements added together have to be equal to 1.

    Input:
    data - data for rescaling
    endmembers - number of endmembers

    Returns:
    data - data after rescaling
    '''

    data = data.view(-1, endmembers).add(1e-7)
    # sum of all elements in single vector
    sum_elements = torch.sum(data, axis=1)
    sum_elements = sum_elements.unsqueeze(1)
    # division of each element by the sum of a single vector
    data = torch.div(data, sum_elements)
    return data

if __name__ == "__main__":
    pass