import math

import pandas as pd
from kde_tools import l2kernel_kde
from GI import gi_l2kernel_kde
from evaluation import MSE
from RACE import count_race_l2
from mLDP_KDE import mldp_kde_l2kernel_kde
from plotting_tools import draw_n_MSE, draw_n_construction_time, draw_n_query_time
from PM import piecewise_l1_l2_kernel_kde
from DM import duchi_l1_l2_kernel_kde
from SW import square_wave_l1_l2_kernel_kde
from parameters import dataset_parameters


def read_file(file):
    data = pd.read_csv(file, sep=',', lineterminator='\n', header=None)
    data = data.values
    return data


const_file_1e4 = "data/big/SYN_10000_50_const.csv"
query_file_1e4 = "data/big/SYN_10000_50_query.csv"
const_data_1e4 = read_file(const_file_1e4)
query_data_1e4 = read_file(query_file_1e4)

const_file_2e4 = "data/big/SYN_20000_50_const.csv"
query_file_2e4 = "data/big/SYN_20000_50_query.csv"
const_data_2e4 = read_file(const_file_2e4)
query_data_2e4 = read_file(query_file_2e4)

const_file_5e4 = "data/big/SYN_50000_50_const.csv"
query_file_5e4 = "data/big/SYN_50000_50_query.csv"
const_data_5e4 = read_file(const_file_5e4)
query_data_5e4 = read_file(query_file_5e4)

const_file_1e5 = "data/big/SYN_100000_50_const.csv"
query_file_1e5 = "data/big/SYN_100000_50_query.csv"
const_data_1e5 = read_file(const_file_1e5)
query_data_1e5 = read_file(query_file_1e5)

const_file_2e5 = "data/big/SYN_200000_50_const.csv"
query_file_2e5 = "data/big/SYN_200000_50_query.csv"
const_data_2e5 = read_file(const_file_2e5)
query_data_2e5 = read_file(query_file_2e5)

const_file_5e5 = "data/big/SYN_500000_50_const.csv"
query_file_5e5 = "data/big/SYN_500000_50_query.csv"
const_data_5e5 = read_file(const_file_5e5)
query_data_5e5 = read_file(query_file_5e5)

const_file_1e6 = "data/big/SYN_1000000_50_const.csv"
query_file_1e6 = "data/big/SYN_1000000_50_query.csv"
const_data_1e6 = read_file(const_file_1e6)
query_data_1e6 = read_file(query_file_1e6)


def choose_dataset(num):
    if num == 1e4:
        return const_data_1e4, query_data_1e4, const_data_1e4.shape[0]
    elif num == 2e4:
        return const_data_2e4, query_data_2e4, const_data_2e4.shape[0]
    elif num == 5e4:
        return const_data_5e4, query_data_5e4, const_data_5e4.shape[0]
    elif num == 1e5:
        return const_data_1e5, query_data_1e5, const_data_1e5.shape[0]
    elif num == 2e5:
        return const_data_2e5, query_data_2e5, const_data_2e5.shape[0]
    elif num == 5e5:
        return const_data_5e5, query_data_5e5, const_data_5e5.shape[0]
    elif num == 1e6:
        return const_data_1e6, query_data_1e6, const_data_1e6.shape[0]


test_n = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
test_epsilon = [1, 5, 20]
nearest_flag = 100
params = dataset_parameters['SYN']
m = params['m']
r_set = params['r_set']
omega = params['omega']
seed_l2lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set = params['L_R_set_for_testN']

''' accurate kde values '''
acc_kde_vals = []
for index, num in enumerate(test_n):
    num = int(num)
    const_data_in_n, query_data_in_n, N = choose_dataset(num)
    acc_kde_vals.append(l2kernel_kde(query_data_in_n, const_data_in_n, N, omega))

''' RACE '''
race_mse = []
race_ctime = []
race_qtime = []
for index, num in enumerate(test_n):
    num = int(num)
    const_data_in_n, query_data_in_n, N = choose_dataset(num)
    race_mse_sum = 0
    race_ctime_sum = 0
    race_qtime_sum = 0
    for temp_seed_l2lsh in seed_l2lsh:
        race_kde_val, c_time, q_time = count_race_l2(query_data_in_n, const_data_in_n, m, omega, N, temp_seed_l2lsh, 1000, 100)
        race_mse_sum += MSE(acc_kde_vals[index], race_kde_val)
        race_ctime_sum += c_time
        race_qtime_sum += q_time
    race_mse.append(race_mse_sum / len(seed_l2lsh))
    race_ctime.append(race_ctime_sum / len(seed_l2lsh))
    race_qtime.append(race_qtime_sum / len(seed_l2lsh))


''' DM-KDE, PM-KDE, SW-KDE, GI-KDE '''
def calc_kde_values(kde_function, epsilon, m, test_num, omega):
    mse_vals = []
    ctime_vals = []
    qtime_vals = []
    for num in test_num:
        const_data_in_n, query_data_in_n, N = choose_dataset(num)
        kde_val, ctime, qtime = kde_function(epsilon, query_data_in_n, const_data_in_n, m, N, omega)
        mse_vals.append(MSE(acc_kde_vals, kde_val))
        ctime_vals.append(ctime)
        qtime_vals.append(qtime)
    return mse_vals, ctime_vals, qtime_vals

# DM-KDE
dm_mse_e_1, _, _ = calc_kde_values(duchi_l1_l2_kernel_kde, 1, m, test_n, omega)
dm_mse_e_5, _, _ = calc_kde_values(duchi_l1_l2_kernel_kde, 5, m, test_n, omega)
dm_mse_e_20, dm_ctime, dm_qtime = calc_kde_values(duchi_l1_l2_kernel_kde, 20, m, test_n, omega)
# PM-KDE
pm_mse_e_1, _, _ = calc_kde_values(piecewise_l1_l2_kernel_kde, 1, m, test_n, omega)
pm_mse_e_5, _, _ = calc_kde_values(piecewise_l1_l2_kernel_kde, 5, m, test_n, omega)
pm_mse_e_20, pm_ctime, pm_qtime = calc_kde_values(piecewise_l1_l2_kernel_kde, 20, m, test_n, omega)
# SW-KDE
sw_mse_e_1, _, _ = calc_kde_values(square_wave_l1_l2_kernel_kde, 1, m, test_n, omega)
sw_mse_e_5, _, _ = calc_kde_values(square_wave_l1_l2_kernel_kde, 5, m, test_n, omega)
sw_mse_e_20, sw_ctime, sw_qtime = calc_kde_values(square_wave_l1_l2_kernel_kde, 20, m, test_n, omega)
# GI-KDE
gi_mse_e_1, _, _ = calc_kde_values(gi_l2kernel_kde, 1, m, test_n, omega)
gi_mse_e_5, _, _ = calc_kde_values(gi_l2kernel_kde, 5, m, test_n, omega)
gi_mse_e_20, gi_ctime, gi_qtime = calc_kde_values(gi_l2kernel_kde, 20, m, test_n, omega)

''' mLDP-KDE '''
mldp_kde_mse_e_1 = []
mldp_kde_ctime_e_1 = []
mldp_kde_qtime_e_1 = []
mldp_kde_mse_e_5 = []
mldp_kde_ctime_e_5 = []
mldp_kde_qtime_e_5 = []
mldp_kde_mse_e_20 = []
mldp_kde_ctime_e_20 = []
mldp_kde_qtime_e_20 = []
for i, e in enumerate(test_epsilon):
    for j, num in enumerate(test_n):
        num = int(num)
        const_data_in_n, query_data_in_n, N = choose_dataset(num)
        mldp_kde_mse_sum = 0
        mldp_kde_ctime_sum = 0
        mldp_kde_qtime_sum = 0
        for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_l2lsh, seed_grr_rehash):
            l2lsh_race_kde, ctime, qtime, _ = mldp_kde_l2kernel_kde(query_data_in_n, e, const_data_in_n, L_R_set[i][0], L_R_set[i][1], m, omega, N,
                                                                   r_set[int(math.log10(nearest_flag))], temp_seed_l2lsh, temp_seed_grr_rehash)
            mldp_kde_mse_sum += MSE(acc_kde_vals[j], l2lsh_race_kde)
            mldp_kde_ctime_sum += ctime
            mldp_kde_qtime_sum += qtime
        globals()[f'mldp_kde_mse_e_{e}'].append(mldp_kde_mse_sum / len(seed_l2lsh))
        globals()[f'mldp_kde_ctime_e_{e}'].append(mldp_kde_ctime_sum / len(seed_l2lsh))
        globals()[f'mldp_kde_qtime_e_{e}'].append(mldp_kde_qtime_sum / len(seed_l2lsh))

gi_mse = [gi_mse_e_1, gi_mse_e_5, gi_mse_e_20]
pm_mse = [pm_mse_e_1, pm_mse_e_5, pm_mse_e_20]
sw_mse = [sw_mse_e_1, sw_mse_e_5, sw_mse_e_20]
dm_mse = [dm_mse_e_1, dm_mse_e_5, dm_mse_e_20]
mldp_kde_mse = [mldp_kde_mse_e_1, mldp_kde_mse_e_5, mldp_kde_mse_e_20]
draw_n_MSE(test_n, race_mse, gi_mse, pm_mse, dm_mse, sw_mse, mldp_kde_mse)
draw_n_construction_time(test_n, race_ctime, gi_ctime, pm_ctime, dm_ctime, sw_ctime, mldp_kde_ctime_e_1, mldp_kde_ctime_e_5, mldp_kde_ctime_e_20)
draw_n_query_time(test_n, race_qtime, gi_qtime, pm_qtime, dm_qtime, sw_qtime, mldp_kde_qtime_e_1, mldp_kde_qtime_e_5, mldp_kde_qtime_e_20)
