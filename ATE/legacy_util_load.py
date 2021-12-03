"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import numpy as np
from torch.utils.data import Dataset

# ----------------------------------------------------------------------------

def spectra_normalisation(X, normalisation=None):
    '''
    Prepare dataset normalisation

    Arguments:
        X - an original dataset
        normalisation - 'minmax': rescaling to the range [-0.5, 0.5]
                        'max': division by the global maximum
                        'max1': division each spectrum by their maximum value
    
    Returns:
        normalised dataset, 
        a dictionary with global minimum and global maximum values
    '''
    global_minimum = np.min(X)
    global_maximum = np.max(X)
    values = {'minimum': global_minimum,
              'maximum': global_maximum}

    if normalisation == 'minmax':
        return (X - global_minimum) / (global_maximum - global_minimum) - 0.5, \
               values
    elif normalisation == 'max':
        return X / global_maximum, values
    elif normalisation == 'max1':
        return np.array([x / np.max(x) for x in X]), values
    else:
        assert False, 'spectra_normalisation(): Bad type!'

# ----------------------------------------------------------------------------

class Hyperspectral(Dataset):

    def __init__(self, fname, normalisation=None):
        self.record = np.load(fname, allow_pickle=True)
        self.cube = np.float32(self.record['data'])
        self.n_endmembers = len(self.record['endmembers'])
        self.endmembers = self.record['endmembers']
        self.abundances = self.record['abundances']
        if len(self.abundances.shape) == 3:
            self.abundances = self.abundances.reshape((-1,
                                                      self.abundances.shape[2]))
        if len(self.cube.shape) == 3:
            self.cube = self.cube.reshape((-1, self.cube.shape[2]))
        self.X = np.reshape(self.cube, (-1, self.cube.shape[-1]))
        if normalisation is not None:
            # Prepare normalisation on the input image and endmembers
            self.X, values = spectra_normalisation(self.X,
                                                   normalisation)
            self.maximum = values['maximum']
            self.minimum = values['minimum']
            self.endmembers, _ = spectra_normalisation(self.endmembers,
                                                       normalisation)
            self.cube = np.reshape(self.X, self.cube.shape)

    def __getitem__(self, index):
        data = self.X[index]
        return data, index

    def __len__(self):
        return len(self.X)

    def get_original(self):
        return self.cube, self.X

    def get_n_endmembers(self):
        return self.n_endmembers

    def get_abundances_gt(self):
        return self.abundances

    def get_endmembers_gt(self):
        return self.endmembers

    def get_global_maximum(self):
        return self.maximum

    def get_global_minimum(self):
        return self.minimum
