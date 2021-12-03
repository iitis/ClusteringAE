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

# -------------------------------------------------------------------

class LossFunction(object):
    """
    General loss function
    """
    def __init__(self,net=None):
        if net is not None:
            self.update(net)

    def set_name(self, name=None):
        self.name = name

    def update(self, net):
        """
        Update internal parameters based on the network architecture
        Parameters
        ----------
        net: architecture
        """
        pass

    def __call__(self, input_tensor, target_tensor):
        """
        Compute the loss function
        Parameters
        ----------
        input_tensor: 2D torch tensor
        input_tensor: 2D torch tensor
        """
        raise NotImplementedError

# -------------------------------------------------------------------

class LossMSE(LossFunction):
    """
    MSE loss
    """
    def __init__(self):
        super(LossMSE, self).set_name('MSE')

    def __call__(self, input_tensor, target_tensor):
        return nn.MSELoss(reduction='mean')(input_tensor, target_tensor)

    def get_name(self):
        return self.name

# -------------------------------------------------------------------

class LossSAD(LossFunction):
    '''
    Spectral Angle Distance (SAD) loss
    '''
    def __init__(self):
        super(LossSAD, self).set_name('SAD')

    def __call__(self, input_tensor, target_tensor):
        # inner product
        dot = torch.sum(input_tensor * target_tensor, dim=1)
        # norm calculations
        image = input_tensor.view(-1, input_tensor.shape[1])
        norm_original = torch.norm(image, p=2, dim=1)
        target = target_tensor.view(-1, target_tensor.shape[1])
        norm_reconstructed = torch.norm(target, p=2, dim=1)
        norm_product = (norm_original.mul(norm_reconstructed)).pow(-1)
        argument = dot.mul(norm_product)
        # for avoiding arccos(1)
        acos = torch.acos(torch.clamp(argument, -1 + 1e-7, 1 - 1e-7))
        loss = torch.mean(acos)

        if torch.isnan(loss):
            raise ValueError(f'Loss is NaN value. Consecutive values - dot: {dot}, \
                norm original: {norm_original}, norm reconstructed: {norm_reconstructed}, \
                    norm product: {norm_product}, argument: {argument}, acos: {acos}, \
                        loss: {loss}, input: {input_tensor}, output: {target}')
        return loss

    def get_name(self):
        return self.name

# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
