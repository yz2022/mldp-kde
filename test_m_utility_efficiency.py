import pandas as pd
from kde_tools import l2kernel_kde
import numpy as np
from GI import gi_l2kernel_kde
from evaluation import MSE
from RACE import count_race_l2
from mLDP_KDE import mldp_kde_l2kernel_kde
from draw_plot import draw_m_MSE, draw_m_query_time, draw_m_construction_time
from PM import piecewise_l1_l2_kernel_kde
from DM import duchi_l1_l2_kernel_kde
from SW import square_wave_l1_l2_kernel_kde
from parameters import dataset_parameters


def read_file(file):
    data = pd.read_csv(file, sep=',', lineterminator='\n', header=None)
    data = data.values
    return data


def calc_r_omega():
    r = []
    omega = []
    for dim in test_m:
        dim = int(dim)
        r.append(0.01 * np.sqrt(dim))
        omega.append(np.sqrt(dim))
    return r, omega


test_m = np.arange(5, 51, 5)
test_epsilon = [1, 20, 50]
# const_file = "data/big/SYN_100000_50_const.csv"
# query_file = "data/big/SYN_100000_50_query.csv"
const_file = "small_datasets/SYN_const.csv"
query_file = "small_datasets/SYN_query.csv"
const_data = read_file(const_file)
query_data = read_file(query_file)
N = const_data.shape[0]
r, omega = calc_r_omega()
params = dataset_parameters['SYN']
seed_l2lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set_e_1 = params['L_R_set_for_testm_e_1']
L_R_set_e_20 = params['L_R_set_for_testm_e_20']
L_R_set_e_50 = params['L_R_set_for_testm_e_50']
L_R_set = [L_R_set_e_1, L_R_set_e_20, L_R_set_e_50]

''' accurate kde values '''
acc_kde_vals = []
for index, m in enumerate(test_m):
    m = int(m)
    query_data_in_m = query_data[:, :m]
    const_data_in_m = const_data[:, :m]
    acc_kde_vals.append(l2kernel_kde(query_data_in_m, const_data_in_m, N, omega[index]))

''' RACE '''
race_mse = []
race_ctime = []
race_qtime = []
for index, m in enumerate(test_m):
    m = int(m)
    query_data_in_m = query_data[:, :m]
    const_data_in_m = const_data[:, :m]
    race_mse_sum = 0
    race_ctime_sum = 0
    race_qtime_sum = 0
    for temp_seed_l2lsh in seed_l2lsh:
        race_kde_val, c_time, q_time = count_race_l2(query_data_in_m, const_data_in_m, m, omega[index], N, temp_seed_l2lsh, 1000, 100)
        race_mse_sum += MSE(acc_kde_vals[index], race_kde_val)
        race_ctime_sum += c_time
        race_qtime_sum += q_time
    race_mse.append(race_mse_sum / len(seed_l2lsh))
    race_ctime.append(race_ctime_sum / len(seed_l2lsh))
    race_qtime.append(race_qtime_sum / len(seed_l2lsh))

''' DM-KDE, PM-KDE, SW-KDE, GI-KDE '''
def calc_kde_values(kde_function, epsilon, N, test_m, omega):
    mse_vals = []
    ctime_vals = []
    qtime_vals = []
    for index, m in enumerate(test_m):
        m = int(m)
        query_data_in_m = query_data[:, :m]
        const_data_in_m = const_data[:, :m]
        kde_val, ctime, qtime = kde_function(epsilon, query_data_in_m, const_data_in_m, m, N, omega[index])
        mse_vals.append(MSE(acc_kde_vals[index], kde_val))
        ctime_vals.append(ctime)
        qtime_vals.append(qtime)
    return mse_vals, ctime_vals, qtime_vals

# DM-KDE
dm_mse_e_1, _, _ = calc_kde_values(duchi_l1_l2_kernel_kde, 1, N, test_m, omega)
dm_mse_e_20, _, _ = calc_kde_values(duchi_l1_l2_kernel_kde, 20, N, test_m, omega)
dm_mse_e_50, dm_ctime, dm_qtime = calc_kde_values(duchi_l1_l2_kernel_kde, 50, N, test_m, omega)
# PM-KDE
pm_mse_e_1, _, _ = calc_kde_values(piecewise_l1_l2_kernel_kde, 1, N, test_m, omega)
pm_mse_e_20, _, _ = calc_kde_values(piecewise_l1_l2_kernel_kde, 20, N, test_m, omega)
pm_mse_e_50, pm_ctime, pm_qtime = calc_kde_values(piecewise_l1_l2_kernel_kde, 50, N, test_m, omega)
# SW-KDE
sw_mse_e_1, _, _ = calc_kde_values(square_wave_l1_l2_kernel_kde, 1, N, test_m, omega)
sw_mse_e_20, _, _ = calc_kde_values(square_wave_l1_l2_kernel_kde, 20, N, test_m, omega)
sw_mse_e_50, sw_ctime, sw_qtime = calc_kde_values(square_wave_l1_l2_kernel_kde, 50, N, test_m, omega)
# GI-KDE
gi_mse_e_1, _, _ = calc_kde_values(gi_l2kernel_kde, 1, N, test_m, omega)
gi_mse_e_20, _, _ = calc_kde_values(gi_l2kernel_kde, 20, N, test_m, omega)
gi_mse_e_50, gi_ctime, gi_qtime = calc_kde_values(gi_l2kernel_kde, 50, N, test_m, omega)

''' mLDP-KDE '''
mldp_kde_mse_e_1 = []
mldp_kde_ctime_e_1 = []
mldp_kde_qtime_e_1 = []
mldp_kde_mse_e_20 = []
mldp_kde_ctime_e_20 = []
mldp_kde_qtime_e_20 = []
mldp_kde_mse_e_50 = []
mldp_kde_ctime_e_50 = []
mldp_kde_qtime_e_50 = []
for i, e in enumerate(test_epsilon):
    for j, m in enumerate(test_m):
        m = int(m)
        query_data_in_m = query_data[:, :m]
        const_data_in_m = const_data[:, :m]
        mldp_kde_mse_sum = 0
        mldp_kde_ctime_sum = 0
        mldp_kde_qtime_sum = 0
        for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_l2lsh, seed_grr_rehash):
            l2lsh_race_kde, ctime, qtime, _ = mldp_kde_l2kernel_kde(query_data_in_m, e, const_data_in_m, L_R_set[i][j][0], L_R_set[i][j][1], m,
                                                                    omega[j], N, r[j], temp_seed_l2lsh, temp_seed_grr_rehash)
            mldp_kde_mse_sum += MSE(acc_kde_vals[j], l2lsh_race_kde)
            mldp_kde_ctime_sum += ctime
            mldp_kde_qtime_sum += qtime
        globals()[f'mldp_kde_mse_e_{e}'].append(mldp_kde_mse_sum / len(seed_l2lsh))
        globals()[f'mldp_kde_ctime_e_{e}'].append(mldp_kde_ctime_sum / len(seed_l2lsh))
        globals()[f'mldp_kde_qtime_e_{e}'].append(mldp_kde_qtime_sum / len(seed_l2lsh))

gi_mse = [gi_mse_e_1, gi_mse_e_20, gi_mse_e_50]
pm_mse = [pm_mse_e_1, pm_mse_e_20, pm_mse_e_50]
sw_mse = [sw_mse_e_1, sw_mse_e_20, sw_mse_e_50]
dm_mse = [dm_mse_e_1, dm_mse_e_20, dm_mse_e_50]
mldp_kde_mse = [mldp_kde_mse_e_1, mldp_kde_mse_e_20, mldp_kde_mse_e_50]
draw_m_MSE(test_m, race_mse, gi_mse, pm_mse, dm_mse, sw_mse, mldp_kde_mse)
draw_m_construction_time(test_m, race_ctime, gi_ctime, pm_ctime, dm_ctime, sw_ctime, mldp_kde_ctime_e_1, mldp_kde_ctime_e_20, mldp_kde_ctime_e_50)
draw_m_query_time(test_m, race_qtime, gi_qtime, pm_qtime, dm_qtime, sw_qtime, mldp_kde_qtime_e_1, mldp_kde_qtime_e_20, mldp_kde_qtime_e_50)

