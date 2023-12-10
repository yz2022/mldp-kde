from draw_plot import draw_epsilon_sketch_communication_twins
import numpy as np
from parameters import dataset_parameters


''' Select dataset '''
datasets = ['CodRNA', 'CovType', 'RCV1', 'Yelp', 'SYN']
selected_flag = 4    # 0: CodRNA, 1:CovType, 2:RCV1, 3: Yelp, 4: SYN

''' Initialize '''
params = dataset_parameters[datasets[selected_flag]]
n = params['n']
m = params['m']
L_R_set = params['L_R_set_by_interval']
epsilon = np.arange(0, 51, 10)
epsilon[0] = 1
n_const = n - 100

''' Get sketch size and communication cost'''
mldp_kde_sketch_size = []
mldp_kde_communication = []

for index, e in enumerate(epsilon):
    L = L_R_set[index][0]
    R = L_R_set[index][1]
    mldp_kde_sketch_size.append((L * R * 8))
    mldp_kde_communication.append(((L * (m + 2) * n_const + L * n_const) * 8) / (1024 * 1024))

draw_epsilon_sketch_communication_twins(epsilon, mldp_kde_sketch_size, mldp_kde_communication, title=datasets[selected_flag])
