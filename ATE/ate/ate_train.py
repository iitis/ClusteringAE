"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import torch
import numpy as np
from torch import nn
from ate.ate_loss import LossMSE

# -----------------------------------------------------------------------------

def train_once(net,
               config,
               data_loader,
               loss_function=LossMSE(),
               optim="adam",
               device='cuda:0'):
    """
    Trains AA once with pure torch

    Parameters
    ----------    
    net: autoencoder object
    config: dict with parameters
    data_loader: data loader object to train the network
    loss_type: type of loss
    optim: type of optimizer
    combined_loss: dictionary with three parameters: 
                   * alpha - coefficient near MSE,
                   * beta - coefficient near SAD,
                   * gamma - coefficient near volume of simplex

    returns: trained network, final loss
    """
    if device == "cuda:0" and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=config["learning_rate"])
    else:
        raise NotImplementedError

    # train net throughout epochs
    report_loss = 0
    net.train()
    for epoch in range(config["no_epochs"]):
        train_loss = 0
        for image, _ in data_loader:
            assert len(image.shape) == 2
            image = image.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, output = net(image)

            loss_function.update(net)
            loss = loss_function(image, output)
            loss.backward()
            ldi = loss.item()
            train_loss += ldi

            optimizer.step()

        report_loss = train_loss/len(data_loader)
        print(f"EPOCH: {epoch}, loss: {report_loss}")
    return net, report_loss

# -----------------------------------------------------------------------------

def weights_initialization(net, method='Kaiming_He_uniform'):
    '''
    Prepare a proper weights initialization in DNN.

    Parameters
    ----------
    method - initialisation method

    Options:
    - kaiming_uniform: He initialization with uniform distribution (default),
    - kaiming_normal: He initialization with normal distribution,
    - xavier_uniform: Xavier (Glorot) initialization with uniform distribution,
    - xavier_normal: Xavier (Glorot) initialization with normal distribution.
    '''
    if method == 'Kaiming_He_uniform':
        # default PyTorch initialization
        pass

    # Xavier initialization
    # "We initialized the biases to be 0" (Glorot, Bengio)
    # He initialization
    # "We also initialize b=0 (He et al.)"
    else:
        # calculate gain in accordance to the activation function
        activation = net.get_activation_function()
        gain = torch.nn.init.calculate_gain(
            nonlinearity=activation['function'],
            param=activation['param']
        )

        for module in net.modules():
            if isinstance(module, nn.Linear):
                if method == 'Kaiming_He_normal':
                # torch documentation:
                # recommended to use only with ''relu'' or ''leaky_relu''
                    a = activation['param'] if activation['function'] == 'leaky_relu' \
                        else 0
                    nn.init.kaiming_normal_(module.weight,
                                            a=a,
                                            nonlinearity=activation['function'])
                elif method == 'Xavier_Glorot_uniform':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif method == 'Xavier_Glorot_normal':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                else:
                    raise NameError('This type of initialization is not implemented!')

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return net

# -----------------------------------------------------------------------------

def weights_modification_decoder(net, device, path=None):
    """
    Modify weights of the decoder.

    Arguments:
      net - PyTorch model for weights modification
      device - device on which 'net' is stored
      path - path for loading decoder weights

    Returns:
      net - PyTorch model after decoder modification
    """
    if path is None:
        return net

    current_dtype = net.linear5.weight.dtype

    pretrained_weights = np.load(path)['endmembers']
    pretrained_weights = torch.nn.Parameter(
        torch.tensor(pretrained_weights,
                     dtype=current_dtype,
                     requires_grad=True,
                     device=device))

    if net.linear5.weight.shape != pretrained_weights.shape:
        raise ValueError('Shape of new endmembers is wrong!')

    net.eval()
    with torch.no_grad():
        net.linear5.weight = pretrained_weights
    return net

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
