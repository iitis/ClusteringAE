"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import os
import random
import numpy as np
import torch

# -------------------------------------------------------------------

def get_device(param_dict={}):
    """
    Returns device, optionally taking into account params (e.g. global_params)
    default: cuda:0

    Parameters
    ----------  
    param_dict: dictionary of parameters with 'device' key (optional)

    Returns
    ----------
    current device

    """
    device = 'cuda:0'
    if 'device' in param_dict:
        assert param_dict['device'] in ['cpu','cuda:0']
        device = param_dict['device']
    if device == 'cuda:0':
        assert torch.cuda.is_available()
    return device    

# -------------------------------------------------------------------

def assert_unique_keys(dic1, dic2, message=""):
    """
    Asserts if two dictionaries have unique keys

    Parameters
    ----------
    dic1: first dictionary
    dic2: second dictionary
    message: message to be printed in exception
    """
    ck = (set(dic1.keys()) & set(dic2.keys()))
    assert len(ck)==0,f"Non-unique keys {message}"

# -------------------------------------------------------------------

def add_new_key(d, k, v):
    """
    Add (key, value) to dict if key not in dict

    Parameters
    ----------    
    d: dictionary to add (key, value) to
    k: key to add if not already present
    v: value of the added key
    """
    if k not in d:
        d[k] = v

# -------------------------------------------------------------------

def save_results(fname, abundances, endmembers):
    """
    save abundances and endmembers to results/{fname}_results.npz
    """
    np.savez(fname+'_results',
             abundances=abundances,
             endmembers=endmembers)

# -------------------------------------------------------------------

def list_architectures(path=None):
    '''
    returns a list of autoencoders in architectures/
    '''
    if path is None:
        path = 'architectures'
    assert os.path.isdir(path),f"No path: {path}"
    files = os.listdir(path)
    return [f.rstrip('.py') for f in files if f.endswith('.py')]

# -------------------------------------------------------------------

def save_model(path, model):
    '''
    Save model's state dict

    Parameters
    ----------
    path: path to save state dict
    model: model object
    '''
    torch.save(model.state_dict(), path)

# -------------------------------------------------------------------

def load_model(path, model, strict=True):
    '''
    Load model's state dict into model object

    Parameters
    ----------
    path: path to state dict to load into model
    model: model object to load state dict into
    strict: whether to load state dict if state dict 
            params do not overlap with model params
    '''
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=strict)
    return model

# -------------------------------------------------------------------

def append_row_to_file(filename, elements):
    '''
    Append a single row to the given file.

    Parameters
    ----------
    filename: folder and name of file
    elements: elements to saving in filename
    '''
    with open(filename, "a+") as stream:
        np.savetxt(stream, np.array(elements)[np.newaxis], delimiter=';', fmt='%s')

# -------------------------------------------------------------------

def save_information_to_file(header, elements, folder, filename=None):
    '''
    Save information about experiments to the file.

    Parameters
    ----------
    header: columns' names
    elements: values of consecutive hyperparameters / metrics
    filename: folder and name of file (default: None)
    '''
    os.makedirs(folder, exist_ok=True)
    if filename is None:
        filename = 'summup_experiments.csv'
    full_path = f'{folder}/{filename}.csv'

    if os.path.isfile(full_path) is False:
        append_row_to_file(full_path, header)

    append_row_to_file(full_path, elements)

# -------------------------------------------------------------------

def print_information(image, output, abundances, ae):
    print(f'Image on input: {image}')
    print(f'Abundances: {abundances}')
    print(f'Image on output: {output}')

    for name, param in ae.named_parameters():
        print(f'name: {name}, {param}, grad: {param.grad}')

# -------------------------------------------------------------------

def set_seed(value):
    '''
    Set deterministic results according to the given value
    (including random, numpy and torch libraries)
    '''
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------

if __name__ == "__main__":
    pass