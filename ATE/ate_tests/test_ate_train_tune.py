"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

#-------------------------------------LOCAL RUN HACK------------------------------
import sys
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#---------------------------------------------------------------------------------

import unittest
from ray import tune

from ate.ate_train_tune import fire_tune
from ate.ate_autoencoders import get_autoencoder
from ate.ate_core import _update_dataset_info
from ate.ate_data import get_dataset
from architectures import basic
from ate_tests.test_params import get_params


class Test(unittest.TestCase):
    def dis_test_train_tune_sanity(self):
        """
        tests if train once runs and returns proper objects
        """
        exp_name = "demo"
        autoencoder_name = "basic"
        dataset_name = "Custom"
        params_global, params_aa = get_params()
        dataset = get_dataset(dataset_name,
                              path=params_global["path_data"],
                              normalisation="max")
        
        _update_dataset_info(params_aa, dataset)
        
        tune_config = {
            "batch_size": 5,
            "learning_rate": 1e-2,
            "weight_decay": 1e-5
        }
        
        best_trained_model,best_config,best_loss,best_checkpoint,result  = fire_tune(autoencoder_name=autoencoder_name,
                                 autoencoder_params=params_aa,
                                 dataset=dataset,
                                 exp_name="tune_test",
                                 tune_config=tune_config)
        self.assertLess(best_loss,10)
        
    def test_train_tune(self):
        """
        tests if train once runs and returns proper objects
        """
        exp_name = "demo"
        autoencoder_name = "basic"
        dataset_name = "Custom"
        params_global,params_aa = get_params()
        dataset = get_dataset(dataset_name,
                              path=params_global["path_data"],
                              normalisation="max")
        
        _update_dataset_info(params_aa, dataset)
        
        config = {
            "l_n": tune.grid_search([10]),
            "batch_size": tune.grid_search([5,400]),
            "learning_rate": tune.grid_search([1e-2]),
            "weight_decay": tune.grid_search([1e-5])
        }

        
        best_trained_model,best_config,best_loss,best_checkpoint,result  = fire_tune(
            autoencoder_name=autoencoder_name,
            grace=3,
            no_epochs=3,
            autoencoder_params=params_aa,
            resources_per_trial={"cpu":3, "gpu":0.25},
            dataset=dataset,
            exp_name="tune_test",
            tune_config=config,
            device='cuda:0')
        
        #print ([l for l in result.results_df['loss']])
        self.assertLess(best_loss,100)
   
            
#---------------------------------------------------------------------------------

if __name__ == "__main__":
    print ("Run tests from ../run_ate_tests.py")