"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import datetime
import matplotlib.pyplot as plt

from sys import platform
from ate.ate_data import get_dataset


# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------

def get_globals(params=None):
    '''
    Add path to SPACE server to dictionary.
    Argument: 'params' dictionary. If is None, 'params' is downloaded
    from exp_demo.py.
    Returns dictionary with an additional key "server_path"
    '''
    if params is None:
        from exp_demo import params_global as params
    params["server_path"] = get_prefix_to_server_path()
    return params

# ---------------------------------------------------------------------------

def compare_endmembers(dataset_name, estimated_endmembers, normalisation,
                       visualisation, path_data, experiment_name):
    '''
    Compare estimated endmembers with ground truth

    Arguments:
        dataset_name - name of the tested dataset
        estimated_endmembers - a vector with endmembers estimated by AE
    '''
    # Compare endmembers
    dataset = get_dataset(dataset_name,
                          path=path_data,
                          normalisation=normalisation)
    endmembers_gt = dataset.get_endmembers_gt()

    plt.title('Comparison of estimated endmembers with ground truth')
    plt.scatter(x=estimated_endmembers[:, 0],
                y=estimated_endmembers[:, 1],
                alpha=0.85,
                label='estimated',
                color='dodgerblue')
    plt.scatter(x=endmembers_gt[:, 0],
                y=endmembers_gt[:, 1],
                alpha=0.85,
                label='ground truth',
                color='orangered')
    plt.legend()
    current_time = datetime.datetime.now().strftime("%d%m%y_%H%M")
    plt.savefig((f'./{visualisation}/exp_'
        f'{experiment_name}_endmembers_'
        f'{current_time}.png'), dpi=400)
    plt.close()

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass