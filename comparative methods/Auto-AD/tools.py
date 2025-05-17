import numpy as np


def hyper_norm(data):
    """
    Normalize the input data according to the specified method, scaling values to [0, 1].
    """

    data = data.astype(np.float64)
    min_val = np.min(data)
    max_val = np.max(data)
    norm = data - min_val
    if max_val == min_val:
        norm = np.zeros(data.shape)
    else:
        norm /= (max_val - min_val)
    return norm
