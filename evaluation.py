import numpy as np


def MSE(v1, v2):
    mse = np.mean((v1 - v2) ** 2)
    return mse
