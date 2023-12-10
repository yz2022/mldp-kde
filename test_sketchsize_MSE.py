import pandas as pd
from draw_plot import draw_sketchsize_MSE
from RACE import count_race_l2
from mLDP_KDE import mldp_kde_l2kernel_kde
from evaluation import MSE
from kde_tools import l2kernel_kde
from parameters import dataset_parameters


''' Select dataset '''
datasets = ['CodRNA', 'CovType', 'RCV1', 'Yelp', 'SYN']
selected_flag = 0  # 0: CodRNA, 1:CovType, 2:RCV1, 3: Yelp, 4: SYN

''' Initialize '''
params = dataset_parameters[datasets[selected_flag]]
m = params['m']
r = params['r']
omega = params['omega']
seed_l2lsh = params['seed_l2lsh']
seed_grr_rehash = params['seed_grr_rehash']
L_R_set_e_1 = params['L_R_set_for_testSketchSize_1']
L_R_set_e_20 = params['L_R_set_for_testSketchSize_20']
L_R_set_e_50 = params['L_R_set_for_testSketchSize_50']
L_R_set = [L_R_set_e_1, L_R_set_e_20, L_R_set_e_50]
L_R_set_race = params['L_R_set_for_testSketchSize_race']
const_file = f"small_datasets/{datasets[selected_flag]}_const.csv"
query_file = f"small_datasets/{datasets[selected_flag]}_query.csv"
const_data = pd.read_csv(const_file, sep=',', lineterminator='\n', header=None)
const_data = const_data.values
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = const_data.shape[0]
epsilon = [1, 20, 50]

''' accurate kde values'''
acc_kde_vals = l2kernel_kde(query_data, const_data, N, omega)

''' RACE '''
race_mse = []
race_LR = []
for comb in L_R_set_race:
    L = comb[0]
    R = comb[1]
    sum = 0
    for temp_seed_l2lsh in seed_l2lsh:
        race_kde_value, _, _ = count_race_l2(query_data, const_data, m, omega, N, temp_seed_l2lsh, L, R)
        sum += MSE(acc_kde_vals, race_kde_value)
    sum /= len(seed_l2lsh)
    race_mse.append(sum)
    race_LR.append(L * R)

''' mLDP-KDE '''
mldp_kde_mse_e_1 = []
mldp_kde_LR_e_1 = []
mldp_kde_mse_e_20 = []
mldp_kde_LR_e_20 = []
mldp_kde_mse_e_50 = []
mldp_kde_LR_e_50 = []
for e in epsilon:
    for comb in globals()[f'L_R_set_e_{e}']:
        L = comb[0]
        R = comb[1]
        sum = 0
        for temp_seed_l2lsh, temp_seed_grr_rehash in zip(seed_l2lsh, seed_grr_rehash):
            mldp_kde_val, _, _, _ = mldp_kde_l2kernel_kde(query_data, e, const_data, L, R, m, omega, N, r, temp_seed_l2lsh, temp_seed_grr_rehash)
            sum = sum + MSE(acc_kde_vals, mldp_kde_val)
        sum = sum / len(seed_l2lsh)
        globals()[f'mldp_kde_mse_e_{e}'].append(sum)
        globals()[f'mldp_kde_LR_e_{e}'].append(L * R)

draw_sketchsize_MSE(mldp_kde_LR_e_1, mldp_kde_LR_e_20, mldp_kde_LR_e_50, race_LR,
                    mldp_kde_mse_e_1, mldp_kde_mse_e_20, mldp_kde_mse_e_50, race_mse, title=datasets[selected_flag])

