import pandas as pd
import numpy as np
from FKM_LL_RACE import fkm_ll_race
from FKM_LR_RACE import fkm_lr_race_kde
from plotting_tools import draw_epsilon_MSE_ang
from PM import piecewise_ang_kernel_kde
from GI import gi_angkernel_kde
from RACE import count_race_ang
from DM import duchi_ang_kernel_kde
from SW import square_wave_ang_kernel_kde
from mLDP_KDE import mldp_kde_angkernel_kde
from evaluation import MSE
from kde_tools import angkernel_kde
from parameters import dataset_parameters


''' Initialize '''
params = dataset_parameters['Yelp']
m = params['m']
seed_anglsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set = params['L_R_set_for_ang']
const_file = "small_datasets/Yelp_const.csv"
query_file = "small_datasets/Yelp_query.csv"
const_data = pd.read_csv(const_file, sep=',', lineterminator='\n', header=None)
const_data = const_data.values
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = const_data.shape[0]
epsilon = np.arange(0, 51, 5)
epsilon[0] = 1

''' Get normalized data '''
l2_norms_const = np.linalg.norm(const_data, axis=1, keepdims=True)
unit_const_data = const_data / l2_norms_const
l2_norms_query = np.linalg.norm(query_data, axis=1, keepdims=True)
unit_query_data = query_data / l2_norms_query

''' accurate kde values'''
acc_kde_vals = angkernel_kde(query_data, const_data, N)

''' RACE '''
race_mse_sum = 0
race_mse = []
for temp_seed_l2lsh in seed_anglsh:
    race_kde_value = count_race_ang(unit_query_data, unit_const_data, m, N, temp_seed_l2lsh, 5000, 2)
    race_mse_sum += MSE(acc_kde_vals, race_kde_value)
race_mse.append(race_mse_sum / len(seed_anglsh))

''' FKM-LL-RACE '''
fkm_ll_race_mse = []
for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    fkm_ll_race_mse_sum = 0
    for temp_seed_l2lsh in seed_anglsh:
        l2lsh_race_kde = fkm_ll_race(unit_query_data, e, unit_const_data, L, R, m, N, temp_seed_l2lsh)
        fkm_ll_race_mse_sum += MSE(acc_kde_vals, l2lsh_race_kde)
    fkm_ll_race_mse.append(fkm_ll_race_mse_sum / len(seed_anglsh))

''' FKM-LR-RACE '''
fkm_lr_race_mse = []
for index, e in enumerate(epsilon):
    L_R_set = [[3, 2], [14, 2], [32, 2], [52, 2], [74, 2], [104, 2], [130, 2], [142, 2], [158, 2], [204, 2], [224, 2]]
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    fkm_lr_race_mse_sum = 0
    for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_anglsh, seed_grr_rehash):
        l2lsh_race_kde = fkm_lr_race_kde(unit_query_data, e, unit_const_data, L, R, m, N, temp_seed_l2lsh, temp_seed_grr_rehash)
        fkm_lr_race_mse_sum += MSE(acc_kde_vals, l2lsh_race_kde)
    fkm_lr_race_mse.append(fkm_lr_race_mse_sum / len(seed_anglsh))

''' DM-KDE, PM-KDE, SW-KDE, GI-KDE '''
def calc_kde_values(epsilon, kde_function, *args):
    mse_vals = []
    for e in epsilon:
        kde_val = kde_function(e, *args)
        mse_vals.append(MSE(acc_kde_vals, kde_val))
    return mse_vals

# DM-KDE
dm_mse = calc_kde_values(epsilon, duchi_ang_kernel_kde, unit_query_data, const_data, m, N)
# PM-KDE
pm_mse = calc_kde_values(epsilon, piecewise_ang_kernel_kde, unit_query_data, const_data, m, N)
# SW-KDE
sw_mse = calc_kde_values(epsilon, square_wave_ang_kernel_kde, unit_query_data, const_data, m, N)
# GI-KDE
gi_mse = calc_kde_values(epsilon, gi_angkernel_kde, unit_query_data, const_data, N)

''' mLDP-KDE '''
mldp_kde_mse = []
for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    mldp_kde_mse_sum = 0
    for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_anglsh, seed_grr_rehash):
        l2lsh_race_kde = mldp_kde_angkernel_kde(unit_query_data, e, unit_const_data, L, R, m, N, temp_seed_l2lsh, temp_seed_grr_rehash)
        mldp_kde_mse_sum += MSE(acc_kde_vals, l2lsh_race_kde)
    mldp_kde_mse.append(mldp_kde_mse_sum / len(seed_anglsh))

draw_epsilon_MSE_ang(epsilon, race_mse, pm_mse, dm_mse, sw_mse, gi_mse, fkm_ll_race_mse, fkm_lr_race_mse, mldp_kde_mse, title="Yelp (Angular)")
