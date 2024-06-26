import numpy as np


dataset_parameters = {
    "CodRNA": {
        "n": 488565,
        "m": 8,
        "r": 0.01,
        "r_set": [0.01, 0.055, 0.1, 0.15, 0.2],
        "r_set_maximum": [0.0864, 0.1470, 0.2888, 0.5105, 0.8494],
        "omega": 0.25,
        "seed_l2lsh": [345, 135, 233, 137, 932, 146, 739, 722, 636, 660],
        "seed_grr_rehash": [338, 77, 457, 183, 76, 679, 761, 872, 969, 895],
        "L_R_set_1nearest": [[14, 12], [16, 12], [34, 12]],  # for epsilon = 1, 5, 20
        "L_R_set_10nearest": [[6, 5], [9, 7], [40, 38]],
        "L_R_set_100nearest": [[6, 5], [8, 6], [9, 6], [10, 9], [10, 7], [12, 9], [18, 16], [18, 16], [18, 16]],  # for epsilon = 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
        "L_R_set_1000nearest": [[5, 4], [8, 6], [12, 8]],
        "L_R_set_10000nearest": [[5, 4], [7, 6], [12, 8]],
        "L_R_set_1nearest_maximum": [[6, 5], [10, 7], [18, 14]],
        "L_R_set_10nearest_maximum": [[6, 5], [9, 7], [18, 16]],
        "L_R_set_100nearest_maximum": [[6, 4], [7, 6], [12, 10]],
        "L_R_set_1000nearest_maximum": [[4, 4], [7, 5], [10, 9]],
        "L_R_set_10000nearest_maximum": [[3, 3], [6, 5], [9, 6]],
        "L_R_set_by_interval": [[6, 5], [9, 6], [10, 7], [18, 16], [18, 16]],
        "L_R_set_for_testSketchSize_1": [[3, 4], [4, 4], [6, 4], [6, 6], [8, 6], [8, 8], [10, 10], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50],
                                         [60, 60], [70, 70]],
        "L_R_set_for_testSketchSize_5": [[3, 4], [3, 6], [4, 6], [6, 6], [8, 6], [10, 10], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60],
                                          [80, 80], [100, 100]],
        "L_R_set_for_testSketchSize_20": [[3, 4], [3, 6], [4, 8], [8, 6], [10, 6], [10, 8], [15, 8], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50],
                                          [60, 60], [80, 80], [100, 100]],
        "L_R_set_for_testSketchSize_race": [[3, 4], [3, 5], [3, 6], [3, 7], [4, 6], [4, 7], [4, 8], [7, 6], [8, 6], [7, 7], [9, 6], [8, 7],
                                            [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [15, 10], [20, 8], [20, 9], [20, 10], [40, 7], [40, 8],
                                            [40, 9], [40, 10], [60, 9], [60, 10], [80, 9], [100, 9], [100, 10], [80, 15], [200, 9], [200, 10],
                                            [200, 15], [400, 10], [300, 25], [400, 20], [800, 10], [900, 10], [500, 20]]
    },
    "CovType": {
        "n": 581012,
        "m": 55,
        "r": 0.01,
        "r_set": [0.01, 0.055, 0.1, 0.3, 0.5],
        "r_set_maximum": [0.1272, 0.2592, 1.4194, 1.4845, 1.7803],
        "omega": 0.5,
        "seed_l2lsh": [345, 915, 291, 137, 134, 146, 274, 919, 636, 660],
        "seed_grr_rehash": [977, 22, 902, 583, 545, 641, 385, 453, 287, 179],
        "L_R_set_1nearest": [[14, 12], [14, 12],[44, 12]],  # for epsilon = 1, 5, 20
        "L_R_set_10nearest": [[10, 8], [16, 9], [30, 10]],
        "L_R_set_100nearest": [[9, 6], [16, 8], [16, 9], [14, 10], [16, 10], [18, 10], [18, 10], [20, 10], [20, 10]],  # for epsilon = 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
        "L_R_set_1000nearest": [[7, 5], [9, 9], [20, 9]],
        "L_R_set_10000nearest": [[3, 5], [9, 8], [16, 9]],
        "L_R_set_1nearest_maximum": [[6, 5], [12, 9], [18, 10]],
        "L_R_set_10nearest_maximum": [[6, 5], [10, 9], [16, 10]],
        "L_R_set_100nearest_maximum": [[6, 5], [9, 9], [16, 10]],
        "L_R_set_1000nearest_maximum": [[3, 4], [2, 6], [10, 9]],
        "L_R_set_10000nearest_maximum": [[2, 3], [2, 6], [2, 10]],
        "L_R_set_by_interval": [[9, 6],[16, 9], [16, 10], [18, 10], [20, 10]],
        "L_R_set_for_testSketchSize_1": [[3, 4], [3, 6], [4, 6], [8, 6], [10, 8], [15, 8], [50, 4], [80, 4], [200, 3], [350, 3], [550, 3],
                                         [60, 60], [80, 70], [100, 90]],
        "L_R_set_for_testSketchSize_5": [[3, 4], [3, 6], [4, 8], [6, 8], [10, 10], [15, 10], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50],
                                          [70, 70], [90, 90]],
        "L_R_set_for_testSketchSize_20": [[3, 4], [3, 6], [3, 8], [4, 10], [6, 10], [10, 10], [15, 10], [30, 10], [20, 20], [30, 30], [40, 40],
                                          [50, 50], [60, 60], [80, 80], [100, 100]],
        "L_R_set_for_testSketchSize_race": [[3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [4, 9], [4, 10], [5, 9], [5, 10],
                                            [7, 9], [7, 10], [8, 9], [8, 10], [9, 10], [10, 10], [15, 10], [20, 10], [20, 15], [40, 10],
                                            [40, 15], [60, 15], [80, 15], [100, 15], [200, 15], [300, 15], [400, 15], [500, 15], [600, 15]]
    },
    "RCV1": {
        "n": 804414,
        "m": 100,
        "r": 0.01,
        "r_set": [0.01, 0.055, 0.2, 0.35, 0.5],
        "r_set_maximum": [0.3000, 0.4476, 0.7694, 0.8034, 0.8565],
        "omega": 0.25,
        "seed_l2lsh": [434, 425, 291, 18, 63, 146, 274, 4, 636, 736],
        "seed_grr_rehash": [991, 463, 92, 816, 429, 76, 553, 231, 954, 385],
        "L_R_set_1nearest": [[12, 12], [14, 12], [30, 12]],  # for epsilon = 1, 5, 20
        "L_R_set_10nearest": [[6, 5], [8, 6], [34, 34]],
        "L_R_set_100nearest": [[4, 4], [6, 6], [7, 6], [8, 7], [9, 8], [8, 7], [9, 7], [14, 14], [14, 14]],  # for epsilon = 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
        "L_R_set_1000nearest": [[4, 4], [7, 6], [9, 7]],
        "L_R_set_10000nearest": [[3, 3], [6, 5], [9, 7]],
        "L_R_set_1nearest_maximum": [[6, 5], [7, 6], [9, 7]],
        "L_R_set_10nearest_maximum": [[6, 5], [6, 5], [9, 6]],
        "L_R_set_100nearest_maximum": [[6, 5], [5, 5], [8, 7]],
        "L_R_set_1000nearest_maximum": [[4, 3], [6, 5], [8, 7]],
        "L_R_set_10000nearest_maximum": [[6, 5], [6, 5], [8, 6]],
        "L_R_set_by_interval": [[4, 4],[7, 6], [9, 8], [9, 7], [14, 14]],
        "L_R_set_for_testSketchSize_1": [[3, 4], [4, 4], [6, 4], [8, 4], [8, 6], [10, 10], [50, 3], [70, 3], [20, 20], [30, 30]],
        "L_R_set_for_testSketchSize_5": [[3, 4], [4, 4], [4, 6], [6, 6], [8, 8], [10, 10], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50],
                                          [70, 70], [100, 100]],
        "L_R_set_for_testSketchSize_20": [[3, 4], [3, 6], [4, 6], [6, 6], [8, 6], [10, 10], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50],
                                          [60, 60], [80, 80], [100, 100]],
        "L_R_set_for_testSketchSize_race": [[3, 4],[3, 5], [3, 6], [3, 7], [4, 6], [4, 7], [5, 6], [5, 7], [6, 6], [6, 7], [8, 6], [7, 7], [8, 7],
                                            [9, 7], [10, 7], [10, 8], [10, 9], [10, 10], [15, 7], [15, 8], [15, 9], [20, 7], [20, 8], [20, 9],
                                            [20, 10], [40, 7], [40, 8], [40, 9], [40, 10], [60, 8], [60, 9], [60, 10], [80, 8], [80, 9], [80, 10],
                                            [100, 9], [100, 10], [100, 15], [200, 9], [200, 10], [300, 9], [300, 10], [400, 9], [400, 10],
                                            [500, 10], [600, 10], [700, 10], [800, 10], [900, 10], [1000, 10]]
    },
    "Yelp": {
        "n": 1986079,
        "m": 100,
        "r": 0.001,
        "r_set": [0.001, 0.00175, 0.0025, 0.00375, 0.005],
        "r_set_maximum": [1.1614, 2.2734, 5.8246, 7.7324, 8.4035],
        "omega": 0.5,
        "seed_l2lsh": [345, 915, 291, 137, 134, 146, 274, 919, 636, 660],
        "seed_grr_rehash": [672, 667, 279, 198, 163, 893, 906, 937, 843, 886],
        "L_R_set_1nearest": [[38, 38], [100, 100], [184, 100]],  # for epsilon = 1, 5, 20
        "L_R_set_10nearest": [[22, 22], [90, 90], [400, 400]],
        "L_R_set_100nearest_2": [[18, 18], [50, 50], [62, 62], [150, 150], [120, 120], [200, 200], [250, 250], [400, 400], [400, 400]],  # for epsilon = 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
        "L_R_set_1000nearest": [[30, 30], [80, 80], [300, 300]],
        "L_R_set_10000nearest": [[20, 20], [52, 52], [250, 250]],
        "L_R_set_1nearest_maximum": [[3, 3], [5, 5], [20, 20]],
        "L_R_set_10nearest_maximum": [[2, 2], [4, 4], [5, 4]],
        "L_R_set_100nearest_maximum": [[2, 2], [3, 3], [10, 10]],
        "L_R_set_1000nearest_maximum": [[2, 2], [2, 2], [10, 10]],
        "L_R_set_10000nearest_maximum": [[2, 2], [3, 3], [4, 4]],
        "L_R_set_by_interval": [[18, 18], [62, 62], [120, 120], [250, 250], [400, 400]],
        "L_R_set_for_testSketchSize_1": [[3, 4], [4, 4], [6, 6], [10, 10], [15, 15], [20, 20], [30, 30], [50, 50], [70, 70], [100, 100], [150, 150],
                                        [200, 200], [300, 300], [500, 500], [700, 700], [1000, 1000]],
        "L_R_set_for_testSketchSize_5": [[4, 3], [3, 6], [8, 6], [10, 10], [15, 15], [20, 20], [40, 40], [50, 50], [70, 70], [100, 100], [150, 150],
                                         [200, 200], [300, 300], [450, 450], [600, 600], [950, 950]],
        "L_R_set_for_testSketchSize_20": [[3, 4], [3, 6], [8, 4], [10, 6], [10, 10], [20, 10], [50, 10], [100, 10], [80, 30], [80, 70], [100, 100],
                                          [150, 150], [200, 200], [300, 300], [400, 400], [600, 600], [1000, 1000]],
        "L_R_set_for_testSketchSize_race": [[3, 4], [4, 4], [3, 6], [4, 6], [6, 6], [8, 6], [10, 6], [10, 8], [15, 6], [10, 10], [15, 8], [15, 10],
                                            [20, 10], [30, 10], [40, 8], [50, 8], [30, 15], [50, 10], [40, 15], [50, 15], [60, 15], [100, 10],
                                            [70, 15], [80, 15], [90, 15], [100, 15], [90, 20], [100, 20], [150, 20], [250, 15], [300, 15], [250, 20],
                                            [350, 15], [400, 15], [450, 15], [350, 20], [400, 20], [450, 20], [500, 20], [400, 30], [650, 20],
                                            [450, 30], [700, 20], [500, 30], [800, 20], [850, 20], [900, 20], [650, 30], [1000, 20], [700, 30],
                                            [750, 30], [800, 30], [850, 30], [900, 30], [950, 30], [950, 60], [1000, 100], [1000, 150], [1000, 250],
                                            [1000, 300], [1000, 350], [1000, 400], [1000, 450], [1000, 500], [1000, 550], [1000, 600], [1000, 650],
                                            [1000, 700], [1000, 750], [1000, 800], [1000, 850], [1000, 900], [1000, 950], [1000, 1000]],
         "L_R_set_for_ang": [[7, 2], [20, 2], [70, 2], [150, 2], [200, 2], [400, 2], [500, 2], [600, 2], [800, 2]],
        "L_R_set_FKMLL_for_ang": [[30, 2], [100, 2], [100, 2], [100, 2], [100, 2], [100, 2], [100, 2], [150, 2], [150, 2]],
        "L_R_set_FKMLR_for_ang": [[4, 2], [9, 2], [22, 2], [40, 2], [60, 2], [80, 2], [100, 2], [150, 2], [150, 2]]
    },
    "SYN": {
        "n": 100000,
        "m": 50,
        "r": 0.01 * np.sqrt(50),
        "r_set": [0.01 * np.sqrt(50), 0.0125 * np.sqrt(50), 0.015 * np.sqrt(50), 0.02 * np.sqrt(50), 0.025 * np.sqrt(50)],
        "r_set_maximum": [0.0871, 0.0932, 0.1019, 0.1151, 11.2314],
        "omega": 1 * np.sqrt(50),
        "seed_l2lsh": [66, 536, 391, 743, 488, 841, 161, 912, 798, 86],
        "seed_grr_rehash": [818, 670, 442, 113, 946, 831, 878, 714, 188, 357],
        "L_R_set_1nearest": [[10, 10], [14, 12], [64, 6]],  # for epsilon = 1, 5, 20
        "L_R_set_10nearest": [[12, 12], [70, 70], [300, 300]],
        "L_R_set_100nearest": [[16, 16], [50, 50], [100, 100], [80, 80], [140, 140], [150, 150], [200, 200], [200, 200], [600, 600]],  # for epsilon = 1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
        "L_R_set_1000nearest": [[12, 12], [46, 46], [180, 180]],
        "L_R_set_10000nearest": [[14, 14], [70, 70], [220, 220]],
        "L_R_set_1nearest_maximum": [[10, 10], [60, 60], [300, 300]],
        "L_R_set_10nearest_maximum": [[16, 16], [60, 60], [300, 300]],
        "L_R_set_100nearest_maximum": [[16, 16], [60, 60], [300, 300]],
        "L_R_set_1000nearest_maximum": [[8, 8], [60, 60], [300, 300]],
        "L_R_set_10000nearest_maximum": [[4, 4], [12, 12], [16, 16]],
        "L_R_set_by_interval": [[16, 16], [100, 100], [140, 140], [200, 200], [600, 600]],
        "L_R_set_for_testSketchSize_1": [[3, 4], [4, 4], [6, 6], [10, 10], [15, 15], [20, 20], [40, 40], [60, 60], [70, 70], [100, 100], [150, 150],
                                         [250, 250], [350, 350], [500, 500], [700, 700], [1000, 1000]],
        "L_R_set_for_testSketchSize_5": [[3, 4], [6, 4], [10, 4], [10, 10], [15, 15], [20, 20], [30, 30], [40, 40], [50, 50], [100, 100], [200, 200],
                                         [300, 300], [350, 350], [500, 500], [750, 750], [1000, 1000]],
        "L_R_set_for_testSketchSize_20": [[3, 4], [6, 4], [10, 4], [30, 4], [50, 4], [50, 10], [50, 20], [60, 50], [100, 100], [150, 150], [200, 200],
                                          [300, 300], [400, 400], [600, 600], [950, 950]],
        "L_R_set_for_testSketchSize_race": [[3, 4], [4, 4], [6, 4], [8, 4], [10, 4], [15, 4], [20, 4], [15, 6], [20, 6], [40, 4], [40, 6], [60, 6],
                                            [80, 6], [100, 6], [200, 6], [300, 6], [400, 6], [500, 6], [600, 6], [700, 6], [800, 6], [900, 6],
                                            [1000, 6], [900, 8], [1000, 8], [1000, 10], [1000, 15], [1000, 20], [1000, 40], [1000, 80], [1000, 100],
                                            [1000, 150], [1000, 200], [1000, 250], [1000, 300], [1000, 350], [1000, 400], [1000, 450], [1000, 500],
                                            [1000, 550], [1000, 600], [1000, 700], [1000, 750], [1000, 800], [1000, 850], [1000, 900], [1000, 950],
                                            [1000, 1000]],
        "L_R_set_for_testN": [[16, 16], [100, 100], [600, 600]],
        "L_R_set_for_testm_e_1": [[26, 26], [18, 18], [14, 14], [24, 24], [10, 10], [24, 24], [24, 24], [16, 16], [14, 14], [26, 26]],
        "L_R_set_for_testm_e_5": [[100, 100], [95, 95], [95, 95], [105, 105], [90, 90], [90, 90], [95, 95], [105, 105], [105, 105], [100, 100]],
        "L_R_set_for_testm_e_20": [[650, 650], [750, 750], [550, 550], [600, 600], [650, 650], [600, 600], [550, 550], [650, 650], [600, 600],
                                   [600, 600]]
    }
}
