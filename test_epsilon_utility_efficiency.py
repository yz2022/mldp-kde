import pandas as pd
import numpy as np
from plotting_tools import draw_epsilon_MSE, draw_epsilon_construction_time, draw_epsilon_query_time
from PM import piecewise_l1_l2_kernel_kde
from GI import gi_l2kernel_kde
from RACE import count_race_l2
from DM import duchi_l1_l2_kernel_kde
from SW import square_wave_l1_l2_kernel_kde
from mLDP_KDE import mldp_kde_l2kernel_kde
from evaluation import MSE
from kde_tools import l2kernel_kde
from parameters import dataset_parameters


''' Select dataset '''
datasets = ['CodRNA', 'CovType', 'RCV1', 'Yelp', 'SYN']
selected_flag = 0    # 0: CodRNA, 1:CovType, 2:RCV1, 3: Yelp, 4: SYN

''' Initialize '''
params = dataset_parameters[datasets[selected_flag]]
r = params['r']
m = params['m']
omega = params['omega']
seed_l2lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set = params['L_R_set']
const_file = f"small_datasets/{datasets[selected_flag]}_const.csv"
query_file = f"small_datasets/{datasets[selected_flag]}_query.csv"
const_data = pd.read_csv(const_file, sep=',', lineterminator='\n', header=None)
const_data = const_data.values
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = const_data.shape[0]
epsilon = np.arange(0, 51, 5)
epsilon[0] = 1

''' accurate kde values'''
acc_kde_vals = l2kernel_kde(query_data, const_data, N, omega)

''' RACE '''
race_mse_sum = 0
race_ctime_sum = 0
race_qtime_sum = 0
race_mse = []
race_ctime = []
race_qtime = []
for temp_seed_l2lsh in seed_l2lsh:
    race_kde_value, ctime, qtime = count_race_l2(query_data, const_data, m, omega, N, temp_seed_l2lsh, 1000, 100)
    race_mse_sum += MSE(acc_kde_vals, race_kde_value)
    race_ctime_sum += ctime
    race_qtime_sum += qtime
race_mse.append(race_mse_sum / len(seed_l2lsh))
race_ctime.append(race_ctime_sum / len(seed_l2lsh))
race_qtime.append(race_qtime_sum / len(seed_l2lsh))

''' DM-KDE, PM-KDE, SW-KDE, GI-KDE '''
def calc_kde_values(epsilon, kde_function, *args):
    mse_vals = []
    ctime_vals = []
    qtime_vals = []
    for e in epsilon:
        kde_val, ctime, qtime = kde_function(e, *args)
        mse_vals.append(MSE(acc_kde_vals, kde_val))
        ctime_vals.append(ctime)
        qtime_vals.append(qtime)
    return mse_vals, ctime_vals, qtime_vals

# DM-KDE
dm_mse, dm_ctime, dm_qtime = calc_kde_values(epsilon, duchi_l1_l2_kernel_kde, query_data, const_data, m, N, omega)
# PM-KDE
pm_mse, pm_ctime, pm_qtime = calc_kde_values(epsilon, piecewise_l1_l2_kernel_kde, query_data, const_data, m, N, omega)
# SW-KDE
sw_mse, sw_ctime, sw_qtime = calc_kde_values(epsilon, square_wave_l1_l2_kernel_kde, query_data, const_data, m, N, omega)
# GI-KDE
gi_mse, gi_ctime, gi_qtime = calc_kde_values(epsilon, gi_l2kernel_kde, query_data, const_data, m, N, omega)

''' mLDP-KDE '''
mldp_kde_mse = []
mldp_kde_ctime = []
mldp_kde_qtime = []
for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    mldp_kde_mse_sum = 0
    mldp_kde_ctime_sum = 0
    mldp_kde_qtime_sum = 0
    for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_l2lsh, seed_grr_rehash):
        l2lsh_race_kde, ctime, qtime, _ = mldp_kde_l2kernel_kde(query_data, e, const_data, L, R, m, omega, N, r, temp_seed_l2lsh,
                                                                temp_seed_grr_rehash)
        mldp_kde_mse_sum += MSE(acc_kde_vals, l2lsh_race_kde)
        mldp_kde_ctime_sum += ctime
        mldp_kde_qtime_sum += qtime
    mldp_kde_mse.append(mldp_kde_mse_sum / len(seed_l2lsh))
    mldp_kde_ctime.append(mldp_kde_ctime_sum / len(seed_l2lsh))
    mldp_kde_qtime.append(mldp_kde_qtime_sum / len(seed_l2lsh))

draw_epsilon_MSE(epsilon, race_mse, pm_mse, dm_mse, sw_mse, gi_mse, mldp_kde_mse, datasets[selected_flag])
draw_epsilon_construction_time(epsilon, mldp_kde_ctime, race_ctime, gi_ctime, pm_ctime, dm_ctime, sw_ctime, datasets[selected_flag])
draw_epsilon_query_time(epsilon, mldp_kde_qtime, race_qtime, gi_qtime, pm_qtime, dm_qtime, sw_qtime, datasets[selected_flag])
