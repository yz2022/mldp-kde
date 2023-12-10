import time
import numpy as np
from scipy.special import comb
from kde_tools import l2kernel_kde, l1kernel_kde, angkernel_kde, rescale, inverse_rescale


''' Function for DM-KDE (l1, l2 kernel)'''
def duchi_l1_l2_kernel_kde(epsilon, query_data, data, dim, n, bandwidth, kernel_flag='l2'):
    start_time = time.perf_counter()
    rescaled_data, data_min, data_max = rescale(data, -1, 1)
    noisy_rescaled_data = []

    if dim % 2 == 1:
        C_d = (2 ** (dim - 1)) / (comb(dim - 1, (dim - 1) / 2))
    else:
        C_d = (2 ** (dim - 1) + comb(dim, dim / 2) / 2) / (comb(dim - 1, dim / 2))
    B = C_d * (np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)

    for d in rescaled_data:
        v = []
        for i in range(dim):
            prob = np.random.uniform(0, 1)
            if prob < 1 / 2 + d[i] / 2:
                v.append(1)
            else:
                v.append(-1)
        u = np.random.random()
        if u < np.exp(epsilon) / (np.exp(epsilon) + 1):
            u = 1
        else:
            u = 0
        while True:
            d_prime = np.random.choice([-B, B], size=dim)
            if u == 1 and np.dot(v, d_prime) >= 0:
                noisy_rescaled_data.append(d_prime)
                break
            elif u == 0 and np.dot(v, d_prime) <= 0:
                noisy_rescaled_data.append(d_prime)
                break
    noisy_data = inverse_rescale(noisy_rescaled_data, data_min, data_max, -1, 1)
    end_time = time.perf_counter()
    const_time = end_time - start_time

    start_time = time.perf_counter()
    dm_kde_val = None
    if kernel_flag == 'l2':
        dm_kde_val = l2kernel_kde(query_data, noisy_data, n, bandwidth)
    elif kernel_flag == 'l1':
        dm_kde_val = l1kernel_kde(query_data, noisy_data, n, bandwidth)
    end_time = time.perf_counter()
    query_time = (end_time - start_time) / (len(query_data))

    return dm_kde_val, const_time, query_time

''' Function for DM-KDE (Angular kernel)'''
def duchi_ang_kernel_kde(epsilon, query_data, data, dim, n):
    rescaled_data, data_min, data_max = rescale(data, -1, 1)
    noisy_rescaled_data = []
    if dim % 2 == 1:
        C_d = (2 ** (dim - 1)) / (comb(dim - 1, (dim - 1) / 2))
    else:
        C_d = (2 ** (dim - 1) + comb(dim, dim / 2) / 2) / (comb(dim - 1, dim / 2))
    B = C_d * (np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)
    for d in rescaled_data:
        v = []
        for i in range(dim):
            prob = np.random.uniform(0, 1)
            if prob < 1 / 2 + d[i] / 2:
                v.append(1)
            else:
                v.append(-1)
        u = np.random.random()
        if u < np.exp(epsilon) / (np.exp(epsilon) + 1):
            u = 1
        else:
            u = 0
        while True:
            d_prime = np.random.choice([-B, B], size=dim)
            if u == 1 and np.dot(v, d_prime) >= 0:
                noisy_rescaled_data.append(d_prime)
                break
            elif u == 0 and np.dot(v, d_prime) <= 0:
                noisy_rescaled_data.append(d_prime)
                break
    noisy_data = inverse_rescale(noisy_rescaled_data, data_min, data_max, -1, 1)
    l2_norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
    unit_noisy_data = noisy_data / l2_norms
    dm_kde_val = angkernel_kde(query_data, unit_noisy_data, n)
    return dm_kde_val
