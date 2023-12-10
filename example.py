import pandas as pd
import numpy as np
import csv
from mLDP_KDE import mldp_kde_l2kernel_kde
from sklearn.datasets import make_blobs
from kde_tools import l2kernel_kde


def sample_data(dim):
    std_variance = 0.01
    X, _ = make_blobs(n_samples=1001, center_box=(-2, 2), centers=2, cluster_std=std_variance, n_features=dim, random_state=42)
    query_data = X[np.random.choice(X.shape[0], size=1, replace=False)]
    const_data = [row for row in X if not any(np.all(row == b_row) for b_row in query_data)]
    np.random.shuffle(const_data)
    const_file = "small_datasets/example/const_data.csv"
    query_file = "small_datasets/example/query_data.csv"
    with open(const_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for d in const_data:
            d = [round(num, 5) for num in d]
            writer.writerow(d)

    with open(query_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for d in query_data:
            d = [round(num, 5) for num in d]
            writer.writerow(d)
    return


np.random.seed(523)
epsilon = 10
d = 0.005
omega = 0.05
dimension = 2
L = 10
R = 8
# sample_data(dimension)
data = pd.read_csv("small_datasets/example/const_data.csv", sep=',', lineterminator='\n', header=None)
data = data.values
query_data = pd.read_csv("small_datasets/example/query_data.csv", sep=',', lineterminator='\n', header=None)
query_data = query_data.values
N = len(data)
seed_l2lsh = np.random.randint(1, 1001)
seed_grr_rehash = np.random.randint(1, 1001)
acc_kde_val = l2kernel_kde(query_data, data, N, omega)
mldp_kde_val, _, _, counts = mldp_kde_l2kernel_kde(query_data, epsilon, data, L, R, dimension, omega, N, d, seed_l2lsh, seed_grr_rehash, flag=0)

''' Print sketch '''
for i, row in enumerate(counts):
    print(i, '| \t', end='')
    for thing in row:
        print(str(int(thing)).rjust(2), end='|')
    print('\n', end='')

