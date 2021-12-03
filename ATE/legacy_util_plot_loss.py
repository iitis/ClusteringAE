"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

from matplotlib import pyplot as plt
import numpy as np


def create_subplot(data, label, alpha, ax=None):
    '''
    plots data on given axis, sets alpha and label to this plot
    '''
    if ax is None:
        ax = plt.gca()
    newdict = {}
    label = eval(label.split('=', 1)[1])
    print('LABEL', label)
    for lk in label:
        splitkey = lk.split('_')
        newkey = ''.join([word[0] for word in splitkey])
        if newkey not in ('bs', 'ne') and label[lk] is not None:
            if newkey == 'cl':
                cl = label[lk]
                newval = (f'\u03B1:{cl["alpha"]}'
                          + f', \u03B2:{cl["beta"]}'
                          + f', \u03B3:{cl["gamma"]}')
            else:
                newval = "{:.2e}".format(label[lk])
        else:
            newval = label[lk]
        newdict[newkey] = newval
    p = ax.plot(np.asarray(data), label=newdict)
    return p


def read_min_losses(file):
    '''
    find the best loss throughout serialized min losses per epoch.
    Read the config and loss
    Return data (== loss), learning rate, l_n and batch size
    '''
    try:
        with open(f"{file}.txt", "r") as g:
            lines = g.readlines()
            data = list(map(float, lines[:-1]))
            config = eval(lines[-1])
            return data, config
    except IOError:
        print("read_min_losses Error opening file."
              + "\nPlease check if the file exists in specified path:",
              f'\n{file}.txt')
    except Exception:
        print("Unknown exception occured")
        print('lines', lines)
        print('data', data)
        print('config', config)


def plot_details(details, main_title, sub_title, output, original=None):
    '''
    create plot for ray tune performance or hyperparameters stability.
    Plot losses, title, subtitle, write to output file
    '''
    plt.gcf().clear()
    _, ax = plt.subplots(figsize=(8, 5))
    alpha = 1.0
    if original:
        '''
        if original axis is given it means the function should plot
        hyperparameters stability.
        It sets the original plot (which is the loss values for the best
        model found by ray tune) as the original axis, the black one.
        Makes all other plots more transparent.
        '''
        ax.plot(np.asarray(original),
                label="ORIGINAL",
                color="black",
                linestyle="dashed",
                linewidth=5)
        alpha = 0.6

    for d in details:
        '''labeled curve of loss values'''
        create_subplot(d['data'], f'{d["label"]}', alpha, ax=ax)

    '''pyplot image labeling'''
    plt.yscale('log')
    plt.ylabel('log of loss')
    plt.xlabel('epochs')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(.5, -0.15),
                    fancybox=True, shadow=True)
    plt.suptitle(main_title)

    plt.title(sub_title)
    # plt.tight_layout()

    plt.savefig(f"{output}", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == '__main__':
    pass
