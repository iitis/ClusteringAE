"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import sys
import shutil
import os
import importlib


#Change this name in accordance with file name!
EXP_ENVIRONMENT_NAME = 'demo'


def ensure_dir(*args):
    '''
    makes sure all needed directories exist

    *args - list of paths to check
    '''
    for arg in args:
        if not os.path.isdir(arg):
            os.mkdir(arg)


def download_dataset(name):
    path = f'data/{name}.npz'
    # If dataset is not downloaded
    if not os.path.isfile(path):
        server_path = importlib.import_module(
            f'exp_{EXP_ENVIRONMENT_NAME}_globals').get_globals()['server_path']
        source = f'{server_path}/{name}.npz'
        # Copy dataset from SPACE to folder 'data'
        if os.path.isfile(source):
            shutil.copy(source, path)
        else:
            raise NameError(f'Such dataset ({name}) is not available! Download it manually and put in /data directory.')


def init_env():
    ensure_dir('results', 'tune',
               os.path.join('tune', 'visualisations'),
               os.path.join('tune', 'checkpoints'),
               os.path.join('tune', 'min_losses'),
               *sys.argv[1:])
    download_dataset('Samson')


if __name__ == "__main__":
    init_env()
