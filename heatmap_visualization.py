import math
import numpy as np
from GI import gi_l2kernel_kde
from RACE import count_race_l2
from mLDP_KDE import mldp_kde_l2kernel_kde
from parameters import dataset_parameters
from sklearn.manifold import TSNE
import pandas as pd
from kde_tools import l2kernel_kde
from plotting_tools import draw_heatmap


''' Select dataset '''
datasets = ['CodRNA', 'CovType', 'RCV1', 'Yelp', 'SYN']
selected_flag = 0  # 0: CodRNA, 1:CovType, 2:RCV1, 3: Yelp, 4: SYN
nearest_flag = 100  # 1: '1'-nearest, 10: '10'-nearest, 100: '100'-nearest, 1000: '1000'-nearest, 10000: '10000'-nearest for choosing r

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
N = const_data.shape[0]
query_data = pd.read_csv(query_file, sep=',', lineterminator='\n', header=None)
query_data = query_data.values
merged_data = np.concatenate((const_data, query_data), axis=0)
epsilon = [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

''' t-SNE embedding '''
tsne = TSNE(n_components=2, random_state=42, verbose=True)
data_embedded = tsne.fit_transform(merged_data)
query_embedded = data_embedded[N:]

''' accurate kde values'''
acc_kde_vals = l2kernel_kde(query_data, const_data, N, omega)
draw_heatmap(query_embedded, acc_kde_vals, acc_kde_vals, datasets, selected_flag, e=1, r=1, method_flag=1)


''' RACE '''
race_kde_value, _, _ = count_race_l2(query_data, const_data, m, omega, N, seed_l2lsh[0], 1000, 100)
draw_heatmap(query_embedded, acc_kde_vals, race_kde_value, datasets, selected_flag, e=1, r=1, method_flag='race')

''' GI '''
for index, e in enumerate(epsilon):
    gi_kde_value, _, _ = gi_l2kernel_kde(e, query_data, const_data, m, N, omega)
    draw_heatmap(query_embedded, acc_kde_vals, gi_kde_value, datasets, selected_flag, e, r=1, method_flag='gi-kde')


''' mLDP-KDE '''
for i, e in enumerate(epsilon):
    L = L_R_set[i][0]
    R = L_R_set[i][1]
    mldp_kde, _, _, _ = mldp_kde_l2kernel_kde(query_data, e, const_data, L, R, m, omega, N, r_set[int(math.log10(nearest_flag))], seed_l2lsh[0],
                                              seed_grr_rehash[0])
    draw_heatmap(query_embedded, acc_kde_vals, mldp_kde, datasets, selected_flag, e, r=r_set[int(math.log10(nearest_flag))], method_flag='mldp-kde')
