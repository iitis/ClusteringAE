"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import json

def read_json(file):
    '''
    Read json file containing loss values and config for torch
    model obtained with ray tune.
    Returns: config, loss
    '''
    config, loss = None, None
    try:
        with open (file, 'r') as jf:
            data = jf.readlines()
    except IOError:
        print("I/O Error. Cannot open or read from file. Check if specified file exists.")
    except:
        print("Unexpected error.")
    else:
        loss = []
        for idx, line in enumerate(data):
            l_data = len(data)
            obj = json.loads(line)
            if idx == l_data-1:
                config = obj["config"]
            loss.append(obj["loss"])
    return config, loss

if __name__ == "__main__":
    conf, l = read_json(
        'ray_results/\
min_losses_more_params_gr_5_adam/\
DEFAULT_a3e28_00000_0_\
batch_size=5,\
l_n=18,\
learning_rate=0.00023132\
_2020-11-13_12-44-51/\
result.json')
    print(conf, l)
    read_json("drgf;jdrgf")