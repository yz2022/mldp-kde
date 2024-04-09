import pandas as pd
from plotting_tools import draw_small_range_epsilon_MSE
from GI import gi_l2kernel_kde
from RACE import count_race_l2
from evaluation import MSE
from kde_tools import l2kernel_kde
from mLDP_KDE import mldp_kde_l2kernel_kde
from parameters import dataset_parameters

''' Select dataset '''
datasets = ['CodRNA', 'CovType', 'RCV1', 'Yelp', 'SYN']
dataset_flag = 3  # 0: CodRNA, 1:CovType, 2:RCV1, 3: Yelp, 4: SYN

''' Initialize '''
params = dataset_parameters[datasets[dataset_flag]]
r_set = params['r_set']
m = params['m']
omega = params['omega']
seed_l2lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set = params[f'L_R_set_100nearest_2']   # choose sketch size parameters
const_file = f"small_datasets/{datasets[dataset_flag]}_const.csv"
query_file = f"small_datasets/{datasets[dataset_flag]}_query.csv"
const_data = pd.read_csv(const_file, sep=',', lineterminator='\n', header=None)
const_data = const_data.values
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = const_data.shape[0]
epsilon = [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

''' accurate kde values'''
acc_kde_vals = l2kernel_kde(query_data, const_data, N, omega)

''' GI '''
def calc_kde_values(epsilon, kde_function, *args):
    mse_vals = []
    for e in epsilon:
        kde_val, ctime, qtime = kde_function(e, *args)
        mse_vals.append(MSE(acc_kde_vals, kde_val))
    return mse_vals

gi_mse = calc_kde_values(epsilon, gi_l2kernel_kde, query_data, const_data, m, N, omega)

''' RACE '''
race_mse_sum = 0
race_mse = []
for temp_seed_l2lsh in seed_l2lsh:
    race_kde_value, _, _ = count_race_l2(query_data, const_data, m, omega, N, temp_seed_l2lsh, 1000, 100)
    race_mse_sum += MSE(acc_kde_vals, race_kde_value)
race_mse.append(race_mse_sum / len(seed_l2lsh))

''' mLDP-KDE '''
mldp_kde_mse = []
for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    mldp_kde_mse_sum = 0
    for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_l2lsh, seed_grr_rehash):
        l2lsh_race_kde, _, _, _ = mldp_kde_l2kernel_kde(query_data, e, const_data, L, R, m, omega, N, r_set[2],
                                                        temp_seed_l2lsh, temp_seed_grr_rehash)
        mldp_kde_mse_sum += MSE(acc_kde_vals, l2lsh_race_kde)
    mldp_kde_mse.append(mldp_kde_mse_sum / len(seed_l2lsh))

draw_small_range_epsilon_MSE(epsilon, race_mse, gi_mse, mldp_kde_mse, datasets[dataset_flag])
