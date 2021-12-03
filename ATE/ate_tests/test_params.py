"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

from sys import platform

def get_params():
    '''
    returns params for tests
    Add path to SPACE server to dictionary.

    Returns:
    params_global,params_aa
    '''

    test_params_global["server_path"] = get_prefix_to_server_path()
    return dict(test_params_global), dict(test_params_aa)

def get_prefix_to_server_path():
    '''
    Get path to SPACE in accordance to operation system.
    Returns prefix of path to SPACE.
    '''
    path_to_images = f'Actual/autoencoders_unmixing/data'

    if platform in ['linux', 'linux2']:
        return f'/run/user/1000/gvfs/smb-share:server=192.168.100.6,share=projects/{path_to_images}'

    elif platform == 'win32':
        return f'//space.ad.iitis.pl/Projects/{path_to_images}'

    else:
        raise SystemError('Your OS is not handled')


test_params_global = {
    'path_data': 'data/',  # dataset dir
    'path_results': 'results',  # path of results
    'path_visualisations': 'visualisations',  # path of visualisations
    'path_tune': 'tune',  # path of tune
    'optim': 'adam',  # optimizer (Adam by default)
    'normalisation': 'max',  # a way of normalisation
    'weights_init': 'Kaiming_He_uniform',  # weights initialization
    'weights_modification': {
        'encoder_path': None,
        'decoder_path': None
    },  # modification of the autoencoder weights
    'seed': None,  # set deterministic results (or None)
}

test_params_aa = {
    "learning_rate": 0.01,
    "no_epochs": 10,
    "weight_decay": 0,
    "batch_size": 5
}


#---------------------------------------------------------------------------------

if __name__ == "__main__":
    pass