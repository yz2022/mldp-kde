import math
import numpy as np
from scipy.special import ndtr
from sklearn.preprocessing import MinMaxScaler


def P_L2(query, data, n, bandwidth):
    real_kde = []
    for q in query:
        val = 0
        for xi in data:
            l2_norm = np.linalg.norm(q - xi)
            try:
                np.seterr(divide='ignore')
                p = 1 - 2 * ndtr(-bandwidth / l2_norm) - 2.0 / (np.sqrt(2 * math.pi) * (bandwidth / l2_norm)) * (
                        1 - np.exp(-0.5 * (bandwidth ** 2) / (l2_norm ** 2)))
            except:
                p = 1
            val += p
        real_kde.append(val / n)
    return real_kde


def P_L1(query, data, n, bandwidth):
    real_kde = []
    for q in query:
        val = 0
        for xi in data:
            l1_norm = np.linalg.norm(q - xi, ord=1)
            if l1_norm == 0:
                p = 1
            else:
                p = (2 / np.pi) * np.arctan(bandwidth / l1_norm) - (l1_norm / (np.pi * bandwidth)) * np.log1p((bandwidth ** 2) / (l1_norm ** 2))
            val += p
        real_kde.append(val / n)
    return real_kde


def P_Ang(query, data, n):
    real_kde = []
    for q in query:
        val = 0
        for xi in data:
            norm1 = np.linalg.norm(q)
            norm2 = np.linalg.norm(xi)
            cosine_similarity = np.dot(q, xi) / (norm1 * norm2)
            theta = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))
            try:
                np.seterr(divide='ignore')
                p = 1 - theta / np.pi
            except:
                p = 1
            val += p
        real_kde.append(val / n)
    return real_kde


def main_P_L2(d, bandwidth):
    main_real_kde = 1 - 2 * ndtr(-bandwidth / d) - 2.0 / (np.sqrt(2 * math.pi) * (bandwidth / d)) * (
            1 - np.exp(-0.5 * (bandwidth ** 2) / (d ** 2)))
    return main_real_kde


def main_P_L1(d, bandwidth):
    main_real_kde = (2 / np.pi) * np.arctan(bandwidth / d) - (d / (np.pi * bandwidth)) * np.log1p(1 + (bandwidth ** 2) / (d ** 2))
    return main_real_kde


def main_P_Ang(d_theta):
    main_real_kde = 1 - d_theta / np.pi
    return main_real_kde


def l2kernel_kde(query, data, n, bandwidth):
    kde_result = np.array(P_L2(query, data, n, bandwidth))
    return kde_result


def l1kernel_kde(query, data, n, bandwidth):
    kde_result = np.array(P_L1(query, data, n, bandwidth))
    return kde_result


def angkernel_kde(query_data, data, n):
    kde_result = np.array(P_Ang(query_data, data, n))
    return kde_result


def rescale(data, left, right):
    scaler = MinMaxScaler(feature_range=(left, right))
    rescaled_data = scaler.fit_transform(data)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return rescaled_data, data_min, data_max


def inverse_rescale(data, data_min, data_max, left, right):
    Result = []
    for xi in data:
        inverse_rescale = []
        for dmax, dmin, d in zip(data_max, data_min, xi):
            inverse_rescale.append(((d - left) / (right - left)) * (dmax - dmin) + dmin)
        Result.append(inverse_rescale)
    return Result
