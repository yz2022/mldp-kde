import csv
import numpy as np
from sklearn.datasets import make_blobs


def sample_data(sample_all, sample_query, data):
    sampled_data = []
    start = 0
    for ind, size in enumerate(sample_all):
        end = start + size
        sub_query = data[start:end]
        sampled_data.extend(
            sub_query[np.random.choice(sub_query.shape[0], size=sample_query[ind], replace=False)])
        start = end
    sampled_data = np.array(sampled_data)
    return sampled_data


def store_data(file, data):
    with open(file, 'w') as csvfile:
        file_writer = csv.writer(csvfile)
        for d in data:
            d = [round(num, 3) for num in d]
            file_writer.writerow(d)


''' Generate SYN datasets '''
np.random.seed(523)

NN = [480000, 161000, 96000, 71000, 41000, 39000, 31000, 28000, 28000, 25000]  # 10^6
sample_10000 = [4800, 1610, 960, 710, 410, 390, 310, 280, 280, 250]
sample_20000 = [9600, 3220, 1920, 1420, 820, 780, 620, 560, 560, 500]
sample_50000 = [24000, 8050, 4800, 3550, 2050, 1950, 1550, 1400, 1400, 1250]
sample_100000 = [48000, 16100, 9600, 7100, 4100, 3900, 3100, 2800, 2800, 2500]
sample_200000 = [96000, 32200, 19200, 14200, 8200, 7800, 6200, 5600, 5600, 5000]
sample_500000 = [240000, 80500, 48000, 35500, 20500, 19500, 15500, 14000, 14000, 12500]

Y, _ = make_blobs(n_samples=NN, center_box=(-2, 2), cluster_std=0.01, n_features=50, random_state=42)

data_10000 = sample_data(NN, sample_10000, Y)
data_20000 = sample_data(NN, sample_20000, Y)
data_50000 = sample_data(NN, sample_50000, Y)
data_100000 = sample_data(NN, sample_100000, Y)
data_200000 = sample_data(NN, sample_200000, Y)
data_500000 = sample_data(NN, sample_500000, Y)

''' Split into construction data and query data '''
sample_100 = [48, 16, 10, 7, 4, 4, 3, 3, 3, 2]

query_data_in_10000 = sample_data(sample_10000, sample_100, data_10000)
const_data_in_10000 = [row for row in data_10000 if not any(np.all(row == b_row) for b_row in query_data_in_10000)]

query_data_in_20000 = sample_data(sample_20000, sample_100, data_20000)
const_data_in_20000 = [row for row in data_20000 if not any(np.all(row == b_row) for b_row in query_data_in_20000)]

query_data_in_50000 = sample_data(sample_50000, sample_100, data_50000)
const_data_in_50000 = [row for row in data_50000 if not any(np.all(row == b_row) for b_row in query_data_in_50000)]

query_data_in_100000 = sample_data(sample_100000, sample_100, data_100000)
const_data_in_100000 = [row for row in data_100000 if not any(np.all(row == b_row) for b_row in query_data_in_100000)]

query_data_in_200000 = sample_data(sample_200000, sample_100, data_200000)
const_data_in_200000 = [row for row in data_200000 if not any(np.all(row == b_row) for b_row in query_data_in_200000)]

query_data_in_500000 = sample_data(sample_500000, sample_100, data_500000)
const_data_in_500000 = [row for row in data_500000 if not any(np.all(row == b_row) for b_row in query_data_in_500000)]

query_data_in_1000000 = sample_data(NN, sample_100, Y)
const_data_in_1000000 = [row for row in Y if not any(np.all(row == b_row) for b_row in query_data_in_1000000)]

''' Save data'''
np.random.shuffle(const_data_in_10000)
np.random.shuffle(const_data_in_20000)
np.random.shuffle(const_data_in_50000)
np.random.shuffle(const_data_in_100000)
np.random.shuffle(const_data_in_200000)
np.random.shuffle(const_data_in_500000)
np.random.shuffle(const_data_in_1000000)
const_file_in_10000 = "data/big/SYN_10000_50_const.csv"
store_data(const_file_in_10000, const_data_in_10000)
const_file_in_20000 = "data/big/SYN_20000_50_const.csv"
store_data(const_file_in_20000, const_data_in_20000)
const_file_in_50000 = "data/big/SYN_50000_50_const.csv"
store_data(const_file_in_50000, const_data_in_50000)
const_file_in_100000 = "data/big/SYN_100000_50_const.csv"
store_data(const_file_in_100000, const_data_in_100000)
const_file_in_200000 = "data/big/SYN_200000_50_const.csv"
store_data(const_file_in_200000, const_data_in_200000)
const_file_in_500000 = "data/big/SYN_500000_50_const.csv"
store_data(const_file_in_500000, const_data_in_500000)
const_file_in_1000000 = "data/big/SYN_1000000_50_const.csv"
store_data(const_file_in_1000000, const_data_in_1000000)

np.random.shuffle(query_data_in_10000)
np.random.shuffle(query_data_in_20000)
np.random.shuffle(query_data_in_50000)
np.random.shuffle(query_data_in_100000)
np.random.shuffle(query_data_in_200000)
np.random.shuffle(query_data_in_500000)
np.random.shuffle(query_data_in_1000000)
query_file_in_10000 = "data/big/SYN_10000_50_query.csv"
store_data(query_file_in_10000, query_data_in_10000)
query_file_in_20000 = "data/big/SYN_20000_50_query.csv"
store_data(query_file_in_20000, query_data_in_20000)
query_file_in_50000 = "data/big/SYN_50000_50_query.csv"
store_data(query_file_in_50000, query_data_in_50000)
query_file_in_100000 = "data/big/SYN_100000_50_query.csv"
store_data(query_file_in_100000, query_data_in_100000)
query_file_in_200000 = "data/big/SYN_200000_50_query.csv"
store_data(query_file_in_200000, query_data_in_200000)
query_file_in_500000 = "data/big/SYN_500000_50_query.csv"
store_data(query_file_in_500000, query_data_in_500000)
query_file_in_1000000 = "data/big/SYN_1000000_50_query.csv"
store_data(query_file_in_1000000, query_data_in_1000000)

np.random.shuffle(Y)
csv_file_Y = "data/big/SYN_1000000_50.csv"
store_data(csv_file_Y, Y)
