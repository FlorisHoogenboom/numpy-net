import numpy as np

def check_2d_array(input):
    if type(input) is not np.ndarray:
        raise TypeError('Input is not an array.')

    if input.ndim != 2:
        raise TypeError('Input is not of the right dimension.')