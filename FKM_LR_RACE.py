import numpy as np
from hashes import ang_lsh, hash_ang_lsh
from RACE import RACE
from mLDP_KDE import inequality_function_ang


def fkm_lr_count_race(anglsh_query, hash_codes, reps, hash_range, n):
    race_kde = []
    S = RACE(n, repetitions=reps, hash_range=hash_range)
    ind = 0
    for d in hash_codes:
        S.main_add(d, ind)
        ind = ind + 1
    for hash_q in anglsh_query:
        race_kde.append(S.query(hash_q))
    result = np.array(race_kde)
    S.clear()
    return result


def binary_search(reps):
    lower_bound = 0.00001
    p = 0.01 / np.pi
    upper_bound = 1 - p - 0.00001
    precision = 0.0000001
    max_iterations = 100000
    guess = (lower_bound + upper_bound) / 2
    for _ in range(max_iterations):
        result = inequality_function_ang(p, guess, reps)
        if abs(result) < precision:
            break
        elif result > 0:
            upper_bound = guess
        else:
            lower_bound = guess
        guess = (lower_bound + upper_bound) / 2
    return guess


def fkm_lr_race_kde(query, epsilon, data, reps, hash_range, dim, n, seed_anglsh, seed_grr_rehash):
    alpha = binary_search(reps)
    gamma = epsilon / (reps * ((0.01 / np.pi) + alpha))
    a = ang_lsh(reps, dim, seed_anglsh)
    hash_values = []
    p_self = np.exp(gamma) / (np.exp(gamma) + reps - 1)
    noisy_datas = []
    for index, d in enumerate(data):
        hash_values_int = np.array(hash_ang_lsh(a, x=d)).astype(int)
        hash_values.append(hash_values_int)
    np.random.seed(seed_grr_rehash)
    for hash_values_int in zip(*hash_values):
        noisy_data = []
        for hash_value in hash_values_int:
            P = np.random.uniform(0, 1)
            if P < p_self:
                noisy_data.append(hash_value)
            else:
                while True:
                    randomNum = np.random.randint(0, hash_range)
                    if randomNum != hash_value:
                        break
                noisy_data.append(randomNum)
        noisy_datas.append(noisy_data)
    hash_query = []
    for index, q in enumerate(query):
        query_hash_value = hash_ang_lsh(a, x=q)
        hash_values_int = np.array(query_hash_value).astype(int)
        hash_query.append(hash_values_int)

    anglsh_race_kde = fkm_lr_count_race(hash_query, noisy_datas, reps, hash_range, n)
    return anglsh_race_kde
