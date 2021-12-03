"""
Autoencoders testing environment (ATE)

Related to the work:
Stable training of autoencoders for hyperspectral unmixing
Paper ID 10040

Source code for the review process of International Conference
on Computer Vision 2021
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat


def process_jasper(dpath, gtpath, n_bands=198, n_rows=100, n_columns=100,
                   n_endmembers=4, save=False, opath=Path("data/Jasper.npz")):
    """
    Processes Jasper Ridge dataset from https://rslab.ut.ac.ir/data (MATLAB version).

    Arguments:
    ----------
        dpath - path to data file.
        gtpath - path to gt file.
        n_bands - number of bands.
        n_rows - number of rown in data matrix.
        n_columns - number of columns in data matrix.
        n_endmembers - number of endmembers.
        save - whether to save the dataset.
        opath - path to output file.
    
    Returns:
    --------
        jasper_data - data matrix of shape (n_rows*n_columns, n_bands).
        jasper_M - endmembers matrix of shape (n_endmembers, n_bands).
        jasper_A - abundances matrix of shape (n_rows*n_columns, n_endmembers).
    """
    jasper_data = loadmat(dpath)["Y"]
    jasper_data = jasper_data.reshape(n_bands, n_columns, n_rows).T.reshape(-1, n_bands)
    jasper_M = loadmat(gtpath)["M"]
    jasper_M = np.swapaxes(jasper_M, 0, -1)
    jasper_A = loadmat(gtpath)["A"]
    jasper_A = jasper_A.reshape(n_endmembers, n_columns, n_rows).T.reshape(-1, n_endmembers)
    if save:
        np.savez(
            opath,
            data=jasper_data,
            endmembers=jasper_M,
            abundances=jasper_A
            )
    return (jasper_data, jasper_M, jasper_A)


if __name__ == '__main__':
    pass
