import math
import numpy as np
import pandas as pd
from evaluation import MSE
from kde_tools import l2kernel_kde
from mLDP_KDE import mldp_kde_l2kernel_kde
from parameters import dataset_parameters

''' Select dataset '''
datasets = ['CodRNA', 'CovType', 'RCV1', 'Yelp', 'SYN']
selected_flag = 0  # 0: CodRNA, 1:CovType, 2:RCV1, 3: Yelp, 4: SYN
nearest_flag = 1  # 1: '1'-nearest, 10: '10'-nearest, 100: '100'-nearest, 1000: '1000'-nearest, 10000: '10000'-nearest for choosing r
''' Initialize '''
params = dataset_parameters[datasets[selected_flag]]
r_set = params['r_set']  # 'r_set': average distance of a point to its {nearest_flag}th nearest neighbor, 'r_set_maximum': maximum distance...
m = params['m']
omega = params['omega']
seed_l2lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set = params[f'L_R_set_{nearest_flag}nearest']  # '~nearest': sketch size corresponding to average distance, '~nearest_maximum': ... maximum distance
const_file = f"small_datasets/{datasets[selected_flag]}_const.csv"
query_file = f"small_datasets/{datasets[selected_flag]}_query.csv"
const_data = pd.read_csv(const_file, sep=',', lineterminator='\n', header=None)
const_data = const_data.values
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = const_data.shape[0]
epsilon = np.arange(0, 51, 5)
epsilon[0] = 1
if len(L_R_set) == 3:
    epsilon = [1, 5, 20]

''' accurate kde values'''
acc_kde_vals = l2kernel_kde(query_data, const_data, N, omega)

''' mLDP-KDE '''
mldp_kde_mse = []
for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    mldp_kde_mse_sum = 0
    for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_l2lsh, seed_grr_rehash):
        l2lsh_race_kde, _, _, _ = mldp_kde_l2kernel_kde(query_data, e, const_data, L, R, m, omega, N, r_set[int(math.log10(nearest_flag))],
                                                        temp_seed_l2lsh, temp_seed_grr_rehash)
        mldp_kde_mse_sum += MSE(acc_kde_vals, l2lsh_race_kde)
    mldp_kde_mse.append(mldp_kde_mse_sum / len(seed_l2lsh))

print(mldp_kde_mse)
