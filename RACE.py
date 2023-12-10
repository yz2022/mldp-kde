import numpy as np
from hashes import l2_lsh, hash_l2_lsh, l1_lsh, hash_l1_lsh, ang_lsh, hash_ang_lsh
import time


''' Class for RACE '''
class RACE:
    def __init__(self, n, repetitions, hash_range):
        self.L = repetitions
        self.R = hash_range
        self.real_counts = np.zeros((self.L, self.R))
        self.N = n

    def add(self, hash_values):
        for idx, hash_value in enumerate(hash_values):
            rehash = int(hash_value)
            rehash = rehash % self.R
            self.real_counts[idx, rehash] += 1

    def main_add(self, hash_values, row_id):
        for hash_value in hash_values:
            self.real_counts[row_id, hash_value] += 1

    def clear(self):
        self.real_counts = np.zeros((self.L, self.R))

    def query(self, hash_values):
        mean = 0
        for idx, hash_value in enumerate(hash_values):
            rehash = int(hash_value)
            rehash = rehash % self.R
            mean = mean + self.real_counts[idx, rehash]
        kde = mean / (self.L * self.N)
        return kde

    def main_query_l1_l2(self, rehash_q, gamma):
        mean = 0
        for idx, rq in enumerate(rehash_q):
            count = self.real_counts[idx, rq % self.R]
            corrected_count = count * ((np.e ** gamma + self.R - 1) / (np.e ** gamma - 1)) - self.N / (np.e ** gamma - 1)
            mean = mean + corrected_count
        kde = mean / (self.L * self.N)
        return kde

    def main_query_ang(self, rehash_q, gamma):
        mean = 0
        for idx, rq in enumerate(rehash_q):
            count = self.real_counts[idx, rq % self.R]
            corrected_count = (count * (np.e ** gamma + 1) - self.N) / (np.e ** gamma - 1)
            mean = mean + corrected_count
        kde = mean / (self.L * self.N)
        return kde

    def print(self):
        for i, row in enumerate(self.real_counts):
            print(i, '| \t', end='')
            for thing in row:
                print(str(int(thing)).rjust(2), end='|')
            print('\n', end='')

    def counts(self):
        return self.real_counts


''' Function for RACE (l2 kernel) '''
def count_race_l2(query, data, dim, bandwidth, n, seed_l2lsh, reps, hash_range, flag=1):
    race_kde = []
    start_time = time.perf_counter()
    S = RACE(n, reps, hash_range)
    W, b = l2_lsh(reps, dim, bandwidth, seed_l2lsh, flag)
    for d in data:
        S.add(hash_l2_lsh(W, b, bandwidth, x=d))
    end_time = time.perf_counter()
    const_time = end_time - start_time
    # S.print()

    start_time = time.perf_counter()
    for q in query:
        race_kde.append(S.query(hash_l2_lsh(W, b, bandwidth, x=q)))
    end_time = time.perf_counter()
    query_time = (end_time - start_time) / (len(query))
    race_kde = np.array(race_kde)
    S.clear()
    return race_kde, const_time, query_time


''' Function for RACE (l1 kernel) '''
def count_race_l1(query, data, dim, bandwidth, n, seed_l1lsh, reps, hash_range):
    race_kde = []
    S = RACE(n, reps, hash_range)
    W, b = l1_lsh(reps, dim, bandwidth, seed_l1lsh)
    for d in data:
        S.add(hash_l1_lsh(W, b, bandwidth, x=d))
    for q in query:
        race_kde.append(S.query(hash_l1_lsh(W, b, bandwidth, x=q)))
    race_kde = np.array(race_kde)
    S.clear()
    return race_kde


''' Function for RACE (angular kernel) '''
def count_race_ang(query, data, dim, n, seed_anglsh, reps, hash_range):
    race_kde = []
    S = RACE(n, reps, hash_range)
    a = ang_lsh(reps, dim, seed_anglsh)
    for d in data:
        S.add(hash_ang_lsh(a, x=d))
    for q in query:
        race_kde.append(S.query(hash_ang_lsh(a, x=q)))
    race_kde = np.array(race_kde)
    S.clear()
    return race_kde
