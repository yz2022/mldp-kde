import numpy as np


def l2_lsh(reps, m, omega, seed_l2lsh, flag=1):
    np.random.seed(seed_l2lsh)
    W = np.random.normal(size=(reps, m))
    if flag == 1:
        np.random.seed(seed_l2lsh)
        b = np.random.uniform(low=0, high=omega, size=reps)
    else:
        b = np.random.uniform(low=0, high=omega, size=reps)
    return W, b


def hash_l2_lsh(W, b, bandwidth, x):
    hash_value = np.floor((np.squeeze(np.dot(W, x)) + b) / bandwidth)
    return hash_value


def l1_lsh(reps, m, bandwidth, seed_l1lsh):
    np.random.seed(seed_l1lsh)
    W = np.random.standard_cauchy(size=(reps, m))
    np.random.seed(seed_l1lsh)
    b = np.random.uniform(low=0, high=bandwidth, size=reps)
    return W, b


def hash_l1_lsh(W, b, bandwidth, x):
    hash_value = np.floor((np.squeeze(np.dot(W, x)) + b) / bandwidth)
    return hash_value


def ang_lsh(L, m, seed_anglsh):
    np.random.seed(seed_anglsh)
    a = np.random.normal(size=(L, m))
    return a


def hash_ang_lsh(a, x):
    hash_value = np.where(np.dot(a, x) < 0, 0, 1)
    return hash_value
