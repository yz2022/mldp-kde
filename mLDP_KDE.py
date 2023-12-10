import numpy as np
from kde_tools import main_P_L2, main_P_L1
from hashes import l2_lsh, hash_l2_lsh, l1_lsh, hash_l1_lsh, ang_lsh, hash_ang_lsh
from RACE import RACE
import time


def map_to_range(num, start, reps):
    mapped_data = []
    for d in num:
        d = d + (-start)
        mapped_data.append(d % reps)
    return mapped_data


def find_max_subarray(nums, reps):
    cnt = {}
    Min = 9999999999
    Max = -Min
    for i in nums:
        if i in cnt:
            cnt[i] += 1
        else:
            cnt[i] = 1
        if i > Max:
            Max = i
        if i < Min:
            Min = i

    subarray_sum = 0
    max_sum = -99999999
    left = 0
    right = 0
    for i in range(Min, Max + 1):
        if (i - Min) < reps:
            if i in cnt:
                subarray_sum = subarray_sum + cnt[i]
            else:
                subarray_sum = subarray_sum + 0
            if subarray_sum > max_sum:
                max_sum = subarray_sum
                left = Min
                right = i

        else:
            if i in cnt:
                subarray_sum = subarray_sum + cnt[i]
            else:
                subarray_sum = subarray_sum + 0
            if (i - reps) in cnt:
                subarray_sum = subarray_sum - cnt[i - reps]
            else:
                subarray_sum = subarray_sum + 0
            if subarray_sum > max_sum:
                max_sum = subarray_sum
                left = i - reps + 1
                right = i
    return left, right


def main_l2kernel_kde(d, bandwidth):
    main_kde = np.array(main_P_L2(d, bandwidth))
    return main_kde


def main_l1kernel_kde(d, bandwidth):
    main_kde = np.array(main_P_L1(d, bandwidth))
    return main_kde


def binary_search_l1_l2(one_kernel, reps, hash_range):
    lower_bound = 0.00001
    upper_bound = 1 - ((hash_range - 1) / hash_range) * one_kernel - 0.00001
    precision = 0.0000001
    max_iterations = 100000
    guess = (lower_bound + upper_bound) / 2

    for _ in range(max_iterations):
        result = inequality_function_l1_l2(one_kernel, guess, reps, hash_range)

        if abs(result) < precision:
            break
        elif result > 0:
            upper_bound = guess
        else:
            lower_bound = guess
        guess = (lower_bound + upper_bound) / 2

    return guess


def binary_search_ang(reps):
    lower_bound = 0.00001
    p = 0.001 / np.pi
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


def inequality_function_l1_l2(one_kernel, alpha, reps, hash_range):
    a = ((hash_range - 1) / hash_range) * one_kernel + alpha
    b = ((hash_range - 1) / hash_range) * one_kernel
    return (a * np.log((a / b)) + (1 - a) * np.log((1 - a) / (1 - b))) - (np.log(10) / reps)


def inequality_function_ang(p, alpha, reps):
    a = p + alpha
    b = p
    return (a * np.log((a / b)) + (1 - a) * np.log((1 - a) / (1 - b))) - (np.log(10) / reps)


def subtract_lists(list1, list2):
    result = [x - y for x, y in zip(list1, list2)]
    return result


def mldp_kde_count_race_l1_l2(l2lsh_query, hash_codes, reps, hash_range, gamma, n, move_list):
    start_time = time.perf_counter()
    mldp_kde_result = []
    S = RACE(n, repetitions=reps, hash_range=hash_range)
    ind = 0
    for d in hash_codes:
        S.main_add(d, ind)
        ind = ind + 1
    # S.print()
    counts = S.counts()
    end_time = time.perf_counter()
    sub_construct_time = end_time - start_time

    start_time = time.perf_counter()
    for hash_q in l2lsh_query:
        rehash_q = subtract_lists(hash_q, move_list)
        mldp_kde_result.append(S.main_query_l1_l2(rehash_q, gamma))
    mldp_kde_result = np.array(mldp_kde_result)
    end_time = time.perf_counter()
    sub_query_time = (end_time - start_time) / (len(l2lsh_query))

    S.clear()
    return mldp_kde_result, sub_construct_time, sub_query_time, counts


def mldp_kde_count_race_ang(l2lsh_query, hash_codes, reps, hash_range, gamma, n):
    mldp_kde_result = []
    S = RACE(n, repetitions=reps, hash_range=hash_range)
    ind = 0
    for d in hash_codes:
        S.main_add(d, ind)
        ind = ind + 1
    for hash_q in l2lsh_query:
        mldp_kde_result.append(S.main_query_ang(hash_q, gamma))
    mldp_kde_result = np.array(mldp_kde_result)
    S.clear()
    return mldp_kde_result


def mldp_kde_l2kernel_kde(query, epsilon, data, reps, hash_range, dim, bandwidth, n, d, seed_l2lsh, seed_grr_rehash, flag=1):
    start_time = time.perf_counter()
    one_kernel = 1 - main_l2kernel_kde(d, bandwidth)
    alpha = binary_search_l1_l2(one_kernel, reps, hash_range)
    temp = 0.8 * d / bandwidth
    gamma = epsilon / (reps * (((hash_range - 1) / hash_range) * temp + alpha))
    W, b = l2_lsh(reps, dim, bandwidth, seed_l2lsh, flag)
    hash_values = []
    p_self = np.exp(gamma) / (np.exp(gamma) + reps - 1)
    noisy_data = []
    for index, d in enumerate(data):
        hash_values_int = np.array(hash_l2_lsh(W, b, bandwidth, x=d)).astype(int)
        hash_values.append(hash_values_int)
    end_time = time.perf_counter()
    const_time_1 = end_time - start_time

    move_list = []
    np.random.seed(seed_grr_rehash)
    start_time = time.perf_counter()
    for hash_values_int in zip(*hash_values):
        left, right = find_max_subarray(hash_values_int, hash_range)
        move_list.append(left)
        rehash_x = map_to_range(hash_values_int, left, hash_range)
        noisy_d = []
        for hash_value in rehash_x:
            P = np.random.uniform(0, 1)
            if P < p_self:
                noisy_d.append(hash_value)
            else:
                while True:
                    random_num = np.random.randint(0, hash_range)
                    if random_num != hash_value:
                        break
                noisy_d.append(random_num)
        noisy_data.append(noisy_d)
    end_time = time.perf_counter()
    construct_time_2 = end_time - start_time

    start_time = time.perf_counter()
    hash_query = []
    for index, q in enumerate(query):
        query_hash_value = hash_l2_lsh(W, b, bandwidth, x=q)
        hash_values_int = np.array(query_hash_value).astype(int)
        hash_query.append(hash_values_int)
    end_time = time.perf_counter()
    query_time = (end_time - start_time) / (len(query))

    l2lsh_race_kde, sub_const_time, sub_query_time, counts = mldp_kde_count_race_l1_l2(hash_query, noisy_data, reps, hash_range, gamma, n, move_list)
    return l2lsh_race_kde, const_time_1 + construct_time_2 + sub_const_time, query_time + sub_query_time, counts


def mldp_kde_l1kernel_kde(query, epsilon, data, reps, hash_range, dim, bandwidth, n, seed_l1lsh, seed_grr_rehash):
    one_kernel = 1 - main_l1kernel_kde(0.05, bandwidth)
    alpha = binary_search_l1_l2(one_kernel, reps, hash_range)
    c1 = 1.2
    c2 = 0.1
    temp1 = (c1 * (hash_range - 1) * 0.05) / (bandwidth * hash_range)
    temp2 = (c2 * (hash_range - 1)) / hash_range
    gamma = epsilon / (reps * (temp1 + temp2 + alpha))
    W, b = l1_lsh(reps, dim, bandwidth, seed_l1lsh)
    hash_values = []
    p_self = np.exp(gamma) / (np.exp(gamma) + reps - 1)
    noisy_data = []
    for index, d in enumerate(data):
        hash_values_int = np.array(hash_l1_lsh(W, b, bandwidth, x=d)).astype(int)
        hash_values.append(hash_values_int)
    move_list = []
    np.random.seed(seed_grr_rehash)
    for hash_values_int in zip(*hash_values):
        left, right = find_max_subarray(hash_values_int, hash_range)
        move_list.append(left)
        rehash_x = map_to_range(hash_values_int, left, hash_range)
        noisy_d = []
        for hash_value in rehash_x:
            P = np.random.uniform(0, 1)
            if P < p_self:
                noisy_d.append(hash_value)
            else:
                while True:
                    random_num = np.random.randint(0, hash_range)
                    if random_num != hash_value:
                        break
                noisy_d.append(random_num)
        noisy_data.append(noisy_d)

    hash_query = []
    for index, q in enumerate(query):
        query_hash_value = hash_l1_lsh(W, b, bandwidth, x=q)
        hash_values_int = np.array(query_hash_value).astype(int)
        hash_query.append(hash_values_int)

    l1lsh_race_kde, _, _, _ = mldp_kde_count_race_l1_l2(hash_query, noisy_data, reps, hash_range, gamma, n, move_list)
    return l1lsh_race_kde


def mldp_kde_angkernel_kde(query, epsilon, data, reps, hash_range, dim, n, seed_anglsh, seed_grr_rehash):
    alpha = binary_search_ang(reps)
    gamma = epsilon / (reps * ((0.001 / np.pi) + alpha))
    a = ang_lsh(reps, dim, seed_anglsh)
    hash_values = []
    p_self = np.exp(gamma) / (np.exp(gamma) + reps - 1)
    noisy_data = []
    for index, d in enumerate(data):
        hash_values_int = np.array(hash_ang_lsh(a, x=d)).astype(int)
        hash_values.append(hash_values_int)
    np.random.seed(seed_grr_rehash)
    for hash_values_int in zip(*hash_values):
        noisy_d = []
        for hash_value in hash_values_int:
            P = np.random.uniform(0, 1)
            if P < p_self:
                noisy_d.append(hash_value)
            else:
                while True:
                    random_num = np.random.randint(0, hash_range)
                    if random_num != hash_value:
                        break
                noisy_d.append(random_num)
        noisy_data.append(noisy_d)

    hash_query = []
    for index, q in enumerate(query):
        query_hash_value = hash_ang_lsh(a, x=q)
        hash_values_int = np.array(query_hash_value).astype(int)
        hash_query.append(hash_values_int)

    anglsh_race_kde = mldp_kde_count_race_ang(hash_query, noisy_data, reps, hash_range, gamma, n)
    return anglsh_race_kde
