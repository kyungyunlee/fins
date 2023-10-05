import scipy.signal 
import numpy as np 
from typing import List

def get_octave_filters():
    """10 octave bandpass filters, each with order 1023
    Return
        firs : shape = (10, 1, 1023)
    """
    f_bounds = []
    f_bounds.append([22.3, 44.5])
    f_bounds.append([44.5, 88.4])
    f_bounds.append([88.4, 176.8])
    f_bounds.append([176.8, 353.6])
    f_bounds.append([353.6, 707.1])
    f_bounds.append([707.1, 1414.2])
    f_bounds.append([1414.2, 2828.4])
    f_bounds.append([2828.4, 5656.8])
    f_bounds.append([5656.8, 11313.6])
    f_bounds.append([11313.6, 22627.2])


    firs: List = []
    for low, high in f_bounds:
        fir = scipy.signal.firwin(1023, np.array([low, high]), pass_zero='bandpass', window='hamming', fs=48000,)
        firs.append(fir)

    firs = np.array(firs)
    firs = np.expand_dims(firs, 1)
    return firs

