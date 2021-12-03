"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import os
import torch


def last_checkpoint(model):
    "returns path of the last checkpoint of a model"
    try:
        with open(os.path.join(model.logdir, "result.json"), "r") as json:
            for cp, _ in enumerate(json):
                pass

    except IOError:
        print('last_checkpoint Input/Ouptut error. \n',
              + 'Make sure both the checkpoint and'
              + 'the output directory exist.\n'
              + f'{os.path.join(model.logdir, "result.json")}')
        return None
    except Exception:
        print("An unexpected exception occured")
        return None

    return os.path.join(model.logdir, f"checkpoint_{cp}", 'checkpoint')


def load_and_save_model(path, output):
    '''load model from checkpoints and save somewhere else.
    path - path of model to load
    name - output path of the saved model'''

    try:
        torch.save(torch.load(path), output)
        print(f"Model saved as {output}")
    except IOError:
        print('load_and_save_model IOError - please make sure that the'
              + ' checkpoint file exists and that the saving function'
              + ' points to an existing directory.\n'
              + path + '\n'
              + output)


def to_file(fname, mode, text):
    """
    Write text to file

    Used to serialize loss values per epoch and
    winner config after ray tune
    """
    path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(path, f'{fname}.txt')
    try:
        with open(filepath, mode) as f:
            f.write(text)
    except IOError:
        print("to_file Error writing to file - IO Exception")
    except Exception:
        print("to_file something went wrong")
