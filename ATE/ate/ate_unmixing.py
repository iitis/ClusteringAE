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

from torch.autograd import Variable
from ate.ate_loss import LossMSE

# -----------------------------------------------------------------------------

def unmix_autoencoder(model,
                      data_loader,
                      n_endmembers,
                      n_bands,
                      n_samples,
                      loss_function=LossMSE(),
                      device='cpu'):
    """
    Similar to autoencoder_train but with model as first parameter

    Parameters
    ----------
    model: nn
    data_loader: data loader instance
    n_endmembers: no endmembers
    n_bands: no bands
    n_samples: no samples
    loss_function: loss function
    device: cpu/cuda:0

    Returns
    ----------
    abundance_image, endmembers_spectra, test_image, total_loss   

    """
    abundance_image = np.zeros((n_samples, n_endmembers), dtype=np.float32)
    test_image = np.zeros((n_samples, n_bands), dtype=np.float32)
    with torch.no_grad():
        model.eval().to(device)
        # get last layer from 'net'
        endmembers = get_endmembers(model)
        loss_table = []
        for img, idx in data_loader:
            assert len(img.shape) == 2
            img = Variable(img).to(device)
            abundances, output = model(img)
            abundance_image[idx] = abundances.cpu().data
            test_image[idx] = output.cpu().data
            loss_function.update(model)
            loss_table.append(loss_function(img, output).cpu().data)
    total_loss = np.sum(loss_table) / len(data_loader.dataset)
    print('Total loss: {}'.format(total_loss))
    endmembers_spectra = endmembers.cpu().data.numpy()  
    return abundance_image, endmembers_spectra, test_image, total_loss

# -------------------------------------------------------------------

def get_endmembers(net):
    '''
    Get endmembers (the last layer) from the network model 

    Parameters
    ----------    
    net: nn to take endmembers from    
    
    Returns
    ----------
    endmembers    
    
    '''
    module = net._modules[f'{list(net._modules)[-1]}']
    endmembers = torch.transpose(module.weight, 0, 1)
    return endmembers

# -------------------------------------------------------------------

if __name__ == "__main__":
    pass
