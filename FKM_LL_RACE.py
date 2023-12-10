import numpy as np
from GI import GeoInd
from RACE import count_race_ang


''' Function for FKM-LL-RACE'''
def fkm_ll_race(query, epsilon, data, reps, hash_range, dim, n, seed_anglsh):
    noisy_data = []
    for d in data:
        geo_ind = GeoInd(epsilon, d, d.shape[0])
        noisy_data.append(geo_ind.draw_point())
    noisy_data = np.array(noisy_data)
    l2_norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
    unit_noisy_data = noisy_data / l2_norms
    geo_race_kde = count_race_ang(query, unit_noisy_data, dim, n, seed_anglsh, reps, hash_range)
    return geo_race_kde
