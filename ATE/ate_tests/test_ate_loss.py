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
from torch import nn
import numpy as np
from torch import from_numpy

from ate_tests.test_params import get_params
from ate.ate_loss import (LossMSE,LossSAD)

def spectral_angle(a,b):
    """
    spectral angle
    """
    va = a / np.sqrt(a.dot(a))
    vb = b / np.sqrt(b.dot(b))
    return np.arccos(np.clip(va.dot(vb), -1, 1))

def msad(X,Y):
    """
    mean spectral angle
    """
    assert len(X)==len(Y)
    #if np.sum(np.abs(X-Y))
    return np.mean([spectral_angle(X[i],Y[i]) for i in range(len(X))])

class Test(unittest.TestCase):
    def test_loss_function_sanity(self):
        for f in [LossMSE(),LossSAD()]:
            a = from_numpy(np.random.rand(10,2)) 
            b = from_numpy(np.random.rand(10,2))
            res = f(a,b).numpy()
            self.assertGreater(res,0)
            if res!=0:
                print (f"test_loss_function_sanity: {res}!=0 for {f}")
        
    def test_loss_function(self):
        for i in range(10):
            A = np.random.rand(10,2)
            B = np.random.rand(10,2)
            if i>7:
                A*=0.001
                B*=0.001
            a = msad(A,B)
            b = LossSAD()(from_numpy(A),from_numpy(B)).numpy()
            if np.abs(a-b)>0.01:
                print ("test_loss_function: SAD, large error for near-zero values!")
            a = nn.MSELoss(reduction='mean')(from_numpy(A),from_numpy(B)).numpy()
            b = LossMSE()(from_numpy(A),from_numpy(B)).numpy()
            self.assertAlmostEqual(a,b,3)

#---------------------------------------------------------------------------------

if __name__ == "__main__":
    print ("Run tests from ../run_ate_tests.py")