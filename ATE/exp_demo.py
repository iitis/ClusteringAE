"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""
import copy

from ate.ate_core import experiment_simple
from run_ate_init import init_env
from ate.ate_loss import LossMSE, LossSAD
from ate.ate_visualise import compare_endmembers
from ate.ate_utils import set_seed

# ---------------------------- PARAMETRISATION----------------------------------

# Change this name in accordance with file name!
EXP_ENVIRONMENT_NAME = 'demo'

params_global = {
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
    },
    'seed': 5,  # set deterministic results (or None)
    'cube_shape': None  # full shape of the given image
}

default_params_aa = {
    "learning_rate": 0.0001,
    "no_epochs": 10,
    "weight_decay": 0,
    "batch_size": 4,
    "l_n": 10
}

# ---------------------------- RUN FUNCTION ------------------------------------

def run():
    """
    Main code to run
    """
    autoencoder_name = 'basic'
    dataset_name = 'Samson'
    experiment_name = 'demo'
    n_runs = 1
    params_aa = default_params_aa
    def_params_global = copy.deepcopy(params_global)
    if dataset_name == 'Samson':
        def_params_global['cube_shape'] = (95, 95, 156)
        def_params_global['weights_modification']['decoder_path'] = '../FCM/Results/Samson_m_2_tol_error_1e-07_max_iter_5000_seed_2.npz'
    elif dataset_name == 'Jasper':
        def_params_global['cube_shape'] = (100, 100, 198)
    set_seed(def_params_global['seed'])

    print("Before", def_params_global)
    print(params_aa)
    evaluation_result, abundance_image, endmembers_spectra, percentage, _, _ = experiment_simple(
        autoencoder_name,
        dataset_name,
        params_aa,
        def_params_global,
        experiment_name=experiment_name,
        mode='train',
        n_runs=n_runs,
        loss_function=LossMSE())
    print(def_params_global)
    # print(f'spectra: {endmembers_spectra}')
    print(f'after: {evaluation_result}')

# -------------------------- RUN DEMO -------------------------------------


if __name__ == "__main__":
    init_env()
    run()
