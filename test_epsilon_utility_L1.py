import pandas as pd
import numpy as np
from plotting_tools import draw_epsilon_MSE_l1 
from PM import piecewise_l1_l2_kernel_kde
from RACE import count_race_l1
from DM import duchi_l1_l2_kernel_kde
from SW import square_wave_l1_l2_kernel_kde
from mLDP_KDE import mldp_kde_l1kernel_kde
from evaluation import MSE
from kde_tools import l1kernel_kde
from parameters import dataset_parameters

''' Initialize '''
params = dataset_parameters['CovType']
m = params['m']
omega = 2.5
seed_l1lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set = [[10, 9], [12, 9], [16, 9], [20, 9], [24, 9], [30, 9], [34, 9], [38, 9], [44, 9], [52, 9], [54, 9]]
const_file = "small_datasets/CovType_const.csv"
query_file = "small_datasets/CovType_query.csv"
const_data = pd.read_csv(const_file, sep=',', lineterminator='\n', header=None)
const_data = const_data.values
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = const_data.shape[0]
epsilon = np.arange(0, 51, 5)
epsilon[0] = 1

''' accurate kde values'''
acc_kde_vals = l1kernel_kde(query_data, const_data, N, omega)

''' RACE '''
race_mse_sum = 0
race_mse = []
for temp_seed_l1lsh in seed_l1lsh:
    race_kde_value = count_race_l1(query_data, const_data, m, omega, N, temp_seed_l1lsh, 1000, 100)
    race_mse_sum += MSE(acc_kde_vals, race_kde_value)
race_mse.append(race_mse_sum / len(seed_l1lsh))

''' DM-KDE, PM-KDE, SW-KDE, GI-KDE '''
def calc_kde_values(epsilon, kde_function, *args, **kwargs):
    mse_vals = []
    for e in epsilon:
        kde_val, _, _ = kde_function(e, *args, **kwargs)
        mse_vals.append(MSE(acc_kde_vals, kde_val))
    return mse_vals

# DM-KDE
dm_mse = calc_kde_values(epsilon, duchi_l1_l2_kernel_kde, query_data, const_data, m, N, omega, kernel_flag='l1')
# PM-KDE
pm_mse = calc_kde_values(epsilon, piecewise_l1_l2_kernel_kde, query_data, const_data, m, N, omega, kernel_flag='l1')
# SW-KDE
sw_mse = calc_kde_values(epsilon, square_wave_l1_l2_kernel_kde, query_data, const_data, m, N, omega, kernel_flag='l1')

''' mLDP-KDE '''
mldp_kde_mse = []
for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    mldp_kde_mse_sum = 0
    for temp_seed_l1lsh, temp_seed_grr_rehash in zip(seed_l1lsh, seed_grr_rehash):
        l1lsh_race_kde = mldp_kde_l1kernel_kde(query_data, e, const_data, L, R, m, omega, N, temp_seed_l1lsh, temp_seed_grr_rehash)
        mldp_kde_mse_sum += MSE(acc_kde_vals, l1lsh_race_kde)
    mldp_kde_mse.append(mldp_kde_mse_sum / len(seed_l1lsh))

draw_epsilon_MSE_l1(epsilon, race_mse, pm_mse, dm_mse, sw_mse, mldp_kde_mse)
