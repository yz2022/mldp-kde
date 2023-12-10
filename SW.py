import numpy as np
import random
from kde_tools import l2kernel_kde, l1kernel_kde, rescale, inverse_rescale, angkernel_kde
import time


''' Function for SW-KDE (l1, l2 kernel)'''
def square_wave_l1_l2_kernel_kde(epsilon, query_data, data, dim, n, bandwidth, kernel_flag='l2'):
    start_time = time.perf_counter()
    rescaled_data, data_min, data_max = rescale(data, 0, 1)
    noisy_rescaled_data = np.copy(rescaled_data)
    eps_row = epsilon / dim
    b = (np.exp(eps_row) * eps_row - np.exp(eps_row) + 1) / (2 * np.exp(eps_row) * (np.exp(eps_row) - 1 - eps_row))
    for row, d in enumerate(rescaled_data):
        for col, d_row_col in enumerate(d):
            choice = np.random.uniform(0, 1)
            low = d_row_col - b
            high = d_row_col + b
            if choice < np.exp(eps_row) * (2 * b - 1) / (2 * b * np.exp(eps_row) + 1):
                noisy_rescaled_data[row][col] = np.random.uniform(low, high)
            else:
                intervals = [[-b, low - 1e-8], [high + 1e-8, 1 + b]]
                noisy_rescaled_data[row][col] = random.uniform(
                    *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0])
    noisy_data = inverse_rescale(noisy_rescaled_data, data_min, data_max, 0, 1)
    end_time = time.perf_counter()
    const_time = end_time - start_time

    start_time = time.perf_counter()
    square_wave_kde_val = None
    if kernel_flag == 'l2':
        square_wave_kde_val = l2kernel_kde(query_data, noisy_data, n, bandwidth)
    elif kernel_flag == 'l1':
        square_wave_kde_val = l1kernel_kde(query_data, noisy_data, n, bandwidth)
    end_time = time.perf_counter()
    query_time = (end_time - start_time) / (len(query_data))
    return square_wave_kde_val, const_time, query_time


''' Function for SW-KDE (l1, l2 kernel)'''
def square_wave_ang_kernel_kde(epsilon, query_data, data, dim, n):
    rescaled_data, data_min, data_max = rescale(data, 0, 1)
    noisy_rescaled_data = np.copy(rescaled_data)
    eps_row = epsilon / dim
    b = (np.exp(eps_row) * eps_row - np.exp(eps_row) + 1) / (2 * np.exp(eps_row) * (np.exp(eps_row) - 1 - eps_row))
    for row, d in enumerate(rescaled_data):
        for col, d_row_col in enumerate(d):
            choice = np.random.uniform(0, 1)
            low = d_row_col - b
            high = d_row_col + b
            if choice < np.exp(eps_row) * (2 * b - 1) / (2 * b * np.exp(eps_row) + 1):
                noisy_rescaled_data[row][col] = np.random.uniform(low, high)
            else:
                intervals = [[-b, low - 1e-8], [high + 1e-8, 1 + b]]
                noisy_rescaled_data[row][col] = random.uniform(
                    *random.choices(intervals, weights=[r[1] - r[0] for r in intervals])[0])
    noisy_data = inverse_rescale(noisy_rescaled_data, data_min, data_max, 0, 1)
    l2_norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
    unit_noisy_data = noisy_data / l2_norms
    square_wave_kde_val = angkernel_kde(query_data, unit_noisy_data, n)
    return square_wave_kde_val
