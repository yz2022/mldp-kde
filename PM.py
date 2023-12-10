from kde_tools import l2kernel_kde, l1kernel_kde, rescale, inverse_rescale, angkernel_kde
import numpy as np
import random
import time


''' Function for PM-KDE (l1, l2 kernel)'''
def piecewise_l1_l2_kernel_kde(epsilon, query_data, data, dim, n, bandwidth, kernel_flag='l2'):
    start_time = time.perf_counter()
    rescaled_data, data_min, data_max = rescale(data, -1, 1)
    noisy_rescaled_data = np.copy(rescaled_data)
    eps_row = epsilon / dim
    C = (np.exp(eps_row / 2) + 1) / (np.exp(eps_row / 2) - 1)
    P = (np.exp(eps_row / 2)) / (np.exp(eps_row / 2) + 1)
    for row, d in enumerate(rescaled_data):
        for col, d_row_col in enumerate(d):
            choice = np.random.uniform(0, 1)
            L = (C + 1) * d_row_col / 2 - (C - 1) / 2
            R = L + C - 1
            intervals = [[-C, L - 1e-8], [R + 1e-8, C]]
            if choice < P:
                noisy_rescaled_data[row][col] = np.random.uniform(L, R)
            else:
                noisy_rescaled_data[row][col] = random.uniform(
                    *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0])
    noisy_data = inverse_rescale(noisy_rescaled_data, data_min, data_max, -1, 1)
    end_time = time.perf_counter()
    const_time = end_time - start_time

    start_time = time.perf_counter()
    piecewise_kde_val = None
    if kernel_flag == 'l2':
        piecewise_kde_val = l2kernel_kde(query_data, noisy_data, n, bandwidth)
    elif kernel_flag == 'l1':
        piecewise_kde_val = l1kernel_kde(query_data, noisy_data, n, bandwidth)
    end_time = time.perf_counter()
    query_time = (end_time - start_time) / (len(query_data))
    return piecewise_kde_val, const_time, query_time


''' Function for PM-KDE (angular kernel)'''
def piecewise_ang_kernel_kde(epsilon, query_data, data, dim, n):
    rescaled_data, data_min, data_max = rescale(data, -1, 1)
    noisy_rescaled_data = np.copy(rescaled_data)
    eps_row = epsilon / dim
    C = (np.exp(eps_row / 2) + 1) / (np.exp(eps_row / 2) - 1)
    P = (np.exp(eps_row / 2)) / (np.exp(eps_row / 2) + 1)
    for row, d in enumerate(rescaled_data):
        for col, d_row_col in enumerate(d):
            choice = np.random.uniform(0, 1)
            L = (C + 1) * d_row_col / 2 - (C - 1) / 2
            R = L + C - 1
            intervals = [[-C, L - 1e-8], [R + 1e-8, C]]
            if choice < P:
                noisy_rescaled_data[row][col] = np.random.uniform(L, R)
            else:
                noisy_rescaled_data[row][col] = random.uniform(
                    *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0])
    noisy_data = inverse_rescale(noisy_rescaled_data, data_min, data_max, -1, 1)
    l2_norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
    unit_noisy_data = noisy_data / l2_norms
    piecewise_kde_val = angkernel_kde(query_data, unit_noisy_data, n)
    return piecewise_kde_val
