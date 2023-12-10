import numpy as np
import time
from kde_tools import l2kernel_kde, angkernel_kde


''' Class for geo-indistinguishability (GI) '''
class GeoInd:
    def __init__(self, epsilon, user_data, dim):
        self.epsilon = epsilon
        self.user_data = user_data
        self.dimensions = dim
        self.noisy_data = np.zeros(dim)

    def draw_point(self):
        x_r = self.perturb_location()
        for i in range(len(self.user_data)):
            self.noisy_data[i] = self.user_data[i] + x_r[i]
        return self.noisy_data

    def perturb_location(self):
        x = np.random.normal(size=self.dimensions)
        sample = self.sample_r()
        x_r = (x / np.linalg.norm(x)) * sample
        return x_r

    def sample_r(self):
        a = self.dimensions
        scale = 1 / self.epsilon
        sample = np.random.gamma(a, scale)
        return sample

    def clear(self):
        self.noisy_data = np.zeros(self.dimensions)


''' Function for GI-KDE (l2 kernel)'''
def gi_l2kernel_kde(epsilon, query, data, m, n, bandwidth):
    start_time = time.perf_counter()
    noisy_data = []
    for d in data:
        geo_ind = GeoInd(epsilon, d, d.shape[0])
        noisy_data.append(geo_ind.draw_point())
    noisy_data = np.array(noisy_data)
    end_time = time.perf_counter()
    const_time = end_time - start_time

    start_time = time.perf_counter()
    geo_kde_val = l2kernel_kde(query, noisy_data, n, bandwidth)
    end_time = time.perf_counter()
    query_time = (end_time - start_time) / len(query)
    return geo_kde_val, const_time, query_time


''' Function for GI-KDE (angular kernel)'''
def gi_angkernel_kde(epsilon, query, data, n):
    noisy_data = []
    for d in data:
        geo_ind = GeoInd(epsilon, d, d.shape[0])
        noisy_data.append(geo_ind.draw_point())
    noisy_data = np.array(noisy_data)
    l2_norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
    unit_noisy_data = noisy_data / l2_norms
    geo_kde_val = angkernel_kde(query, unit_noisy_data, n)
    return geo_kde_val

