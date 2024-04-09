import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import median_filter


def draw_epsilon_MSE(epsilon, race_mse, pm_mse, dm_mse, sw_mse, gi_mse, mldp_kde_mse, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.set_yscale('log')
    ax.set_xlim(-2, 53)

    ylim_dict = {
        "CodRNA": (0.000005, 0.1),
        "CovType": (0.000001, 0.05),
        "RCV1": (0.000001, 0.1),
        "Yelp": (0.000005, 5),
        "SYN": (0.00001, 1)
    }
    if title in ylim_dict:
        ax.set_ylim(*ylim_dict[title])
        ax.yaxis.set_major_locator(plt.FixedLocator([10 ** i for i in range(-6, 2)]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 2) for j in range(2, 10)]))

    plt.plot(epsilon, [race_mse] * len(epsilon), linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False,
             color='black', label="RACE")
    plt.plot(epsilon, dm_mse, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='purple', label="DM-KDE")
    plt.plot(epsilon, pm_mse, linestyle='-', marker='s', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='green', label="PM-KDE")
    plt.plot(epsilon, sw_mse, linestyle='-', marker='D', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='orange', label="SW-KDE")
    plt.plot(epsilon, gi_mse, linestyle='-', marker='x', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='blue', label="GI-KDE")
    plt.plot(epsilon, mldp_kde_mse, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='red', label="mLDP-KDE")

    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(title, fontsize=30)
    plt.show()


def draw_small_range_epsilon_MSE(epsilon, race_mse, gi_mse, mldp_kde_mse, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]))
    ax.set_yscale('log')
    ax.set_xlim(0, 21)
    ax.set_xticklabels(['1', '', '5', '', '10', '', '15', '', '20'])
    ylim_dict = {
        "CodRNA": (0.000005, 0.1),
        "CovType": (0.000001, 0.05),
        "RCV1": (0.000001, 0.1),
        "Yelp": (0.000005, 5),
        "SYN": (0.00001, 1)
    }
    if title in ylim_dict:
        ax.set_ylim(*ylim_dict[title])
        ax.yaxis.set_major_locator(plt.FixedLocator([10 ** i for i in range(-6, 2)]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 2) for j in range(2, 10)]))
    plt.plot(epsilon, [race_mse] * len(epsilon), linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False,
             color='black', label="RACE")
    plt.plot(epsilon, gi_mse, linestyle='-', marker='x', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='blue', label="GI-KDE")
    plt.plot(epsilon, mldp_kde_mse, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='red', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(title, fontsize=30)
    plt.show()


def draw_epsilon_MSE_l1(epsilon, race_mse, pm_mse, dm_mse, sw_mse, mldp_kde_mse):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.set_yscale('log')
    ax.set_xlim(-2, 53)
    ax.set_ylim(0.000005, 0.1)
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 0) for j in range(2, 10)]))
    plt.plot(epsilon, [race_mse] * len(epsilon), linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.plot(epsilon, dm_mse, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='purple', label="DM-KDE")
    plt.plot(epsilon, pm_mse, linestyle='-', marker='s', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='green', label="PM-KDE")
    plt.plot(epsilon, sw_mse, linestyle='-', marker='D', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='orange', label="SW-KDE")
    plt.plot(epsilon, mldp_kde_mse, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='red', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(r"CovType ($l_1$-LSH)", fontsize=30)
    plt.show()


def draw_epsilon_MSE_ang(epsilon, race_mse, pm_mse, dm_mse, sw_mse, gi_mse, fkm_ll_race_mse, fkm_lr_race_mse, mldp_kde_mse, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.set_yscale('log')
    ax.set_xlim(-2, 53)
    ax.set_ylim(1e-6, 1e-2)
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, -1) for j in range(2, 10)]))
    plt.plot(epsilon, [race_mse] * len(epsilon), linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.plot(epsilon, dm_mse, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='purple', label="DM-KDE")
    plt.plot(epsilon, pm_mse, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='green', label="PM-KDE")
    plt.plot(epsilon, sw_mse, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='orange', label="SW-KDE")
    plt.plot(epsilon, gi_mse, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='blue', label="GI-KDE")
    plt.plot(epsilon, fkm_ll_race_mse, linestyle='-', marker='v', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='grey', label="FKM-LL-RACE")
    plt.plot(epsilon, fkm_lr_race_mse, linestyle='-', marker='.', linewidth=1, markersize=15, markeredgewidth=3,
             clip_on=False, color='c', label="FKM-LR-RACE")
    plt.plot(epsilon, mldp_kde_mse, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='red', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(title, fontsize=30)
    plt.show()


def draw_epsilon_construction_time(epsilon, mldp_kde_ctime, race_ctime, gi_ctime, pm_ctime, dm_ctime, sw_ctime, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.set_yscale('log')
    ax.set_xlim(-2, 53)

    ylim_dict = {
        "CodRNA": (1, 5000),
        "CovType": (5, 1000),
        "RCV1": (10, 10000),
        "Yelp": (10, 10000),
        "SYN": (1, 100)
    }
    if title in ylim_dict:
        ax.set_ylim(*ylim_dict[title])
        ax.yaxis.set_major_locator(plt.FixedLocator([10 ** i for i in range(0, 5)]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(0, 5) for j in range(2, 10)]))

    plt.plot(epsilon, [race_ctime] * len(epsilon), linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False,
             color='black', label="RACE")
    plt.plot(epsilon, dm_ctime, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='purple', label="DM-KDE")
    plt.plot(epsilon, pm_ctime, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='green', label="PM-KDE")
    plt.plot(epsilon, sw_ctime, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='orange', label="SW-KDE")
    plt.plot(epsilon, gi_ctime, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='blue', label="GI-KDE")
    plt.plot(epsilon, mldp_kde_ctime, linestyle='-', marker='o', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='red', label="mLDP-KDE")

    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=35)
    plt.ylabel(r'Construction Time (s)')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(title, fontsize=30)
    plt.show()


def draw_epsilon_query_time(epsilon, mldp_kde_qtime, race_qtime, gi_qtime, pm_qtime, dm_qtime, sw_qtime, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.set_yscale('log')
    ax.set_xlim(-2, 53)
    ax.set_ylim(1e-6, 1e2)
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-6, 1e-4, 1e-2, 1e0, 1e2]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 3) for j in range(2, 10)]))

    plt.plot(epsilon, [race_qtime] * len(epsilon), linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False,
             color='black', label="RACE")
    plt.plot(epsilon, dm_qtime, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='purple', label="DM-KDE")
    plt.plot(epsilon, pm_qtime, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='green', label="PM-KDE")
    plt.plot(epsilon, sw_qtime, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='orange', label="SW-KDE")
    plt.plot(epsilon, gi_qtime, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none',
             markeredgewidth=3, clip_on=False, color='blue', label="GI-KDE")
    plt.plot(epsilon, mldp_kde_qtime, linestyle='-', marker='o', linewidth=1, markersize=15,
             markerfacecolor='none', markeredgewidth=3, clip_on=False, color='red', label="mLDP-KDE")

    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('Query Time (s)')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(title, fontsize=30)
    plt.show()


def draw_epsilon_sketch_communication_twins(epsilon, mldp_kde_sketch_size, mldp_kde_communication, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 35,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })

    fig, ax1 = plt.subplots(figsize=(6.4, 5.7))
    ax1.set_xlabel(r'$\varepsilon$', fontname='Times New Roman', fontsize=40)
    ax1.set_ylabel(r'Sketch Size (B)', color='red')
    ax1.set_yscale('log')
    ax1.set_xlim(-5, 56)
    ax1_ylim_dict = {
        "CodRNA": (1e2, 1e4),
        "CovType": (1e2, 1e5),
        "RCV1": (1e3, 1e4),
        "Yelp": (1e4, 1e6),
        "SYN": (1e2, 1e5)
    }
    if title in ax1_ylim_dict:
        ax1.set_ylim(*ax1_ylim_dict[title])
        ax1.yaxis.set_major_locator(plt.FixedLocator([10 ** i for i in range(2, 7)]))
        ax1.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(2, 7) for j in range(2, 10)]))
    ax1.xaxis.set_major_locator(plt.FixedLocator([1, 10, 20, 30, 40, 50]))

    ax1.bar(epsilon - 1.75, mldp_kde_sketch_size, color='red', width=3.5, label="mLDP-KDE")
    ax1.tick_params(axis='x', labelsize=30)
    ax1.tick_params(axis='y', labelcolor='red', labelsize=30)

    ax2 = ax1.twinx()
    ax2.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: ''))
    ax2.set_ylabel('Communication (MB)', color='blue')
    ax2.set_yscale('log')
    ax2_ylim_dict = {
        "CodRNA": (1e2, 1e4),
        "CovType": (1e3, 1e5),
        "RCV1": (1e3, 1e5),
        "Yelp": (1e4, 1e7),
        "SYN": (1e2, 1e5)
    }
    if title in ax2_ylim_dict:
        ax2.set_ylim(*ax2_ylim_dict[title])
        ax2.yaxis.set_major_locator(plt.FixedLocator([10 ** i for i in range(2, 8)]))
        ax2.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(2, 8) for j in range(2, 10)]))

    ax2.bar(epsilon + 1.75, mldp_kde_communication, color='blue', width=3.5, label="mLDP-KDE")
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=30)
    plt.title(title, fontsize=35)
    plt.show()


def draw_sketchsize_MSE(x_values_e_1, x_values_e_20, x_values_e_50, x_values_race, y_values_e_1, y_values_e_20, y_values_e_50, y_values_race, title):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(plt.FixedLocator([10, 100, 1000, 10000, 100000]))
    ax.xaxis.set_minor_locator(plt.FixedLocator([20, 50, 200, 500, 2000, 5000, 20000, 50000]))

    ylim_dict = {
        "CodRNA": (1e-5, 1e-1),
        "CovType": (5e-6, 1e-1),
        "RCV1": (5e-6, 1e-1),
        "Yelp": (5e-6, 5e-2),
        "SYN": (1e-5, 1e-1)
    }
    if title in ylim_dict:
        ax.set_ylim(*ylim_dict[title])
        ax.yaxis.set_major_locator(plt.FixedLocator([10 ** i for i in range(-6, 0)]))
        ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 0) for j in range(2, 10)]))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: ''))

    plt.plot(x_values_race, y_values_race, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.scatter(x_values_e_1, y_values_e_1, marker='x', linewidth=3, s=200, linewidths=2, clip_on=False, color='indianred', label="mLDP-KDE")
    plt.scatter(x_values_e_20, y_values_e_20, marker='o', linewidth=3, s=200, linewidths=2, clip_on=False, color='red', label="mLDP-KDE")
    plt.scatter(x_values_e_50, y_values_e_50, marker='^', linewidth=3, s=200, linewidths=2, clip_on=False, color='darkred', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$L \times R$', fontname='Times New Roman', fontsize=30)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(title, fontsize=30)
    plt.show()


def draw_n_MSE(test_num, race_mse, gi_mse, pm_mse, dm_mse, sw_mse, mldp_kde_mse):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    test_epsilon = [1, 20, 50]
    for index, e in enumerate(test_epsilon):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xscale('log')
        min = 1e4 - 2e3 - 3e2
        max = 1e6 + 3e5
        ax.set_xlim(min, max)
        ax.xaxis.set_major_locator(plt.FixedLocator([1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]))
        ax.xaxis.set_minor_locator(plt.FixedLocator([]))
        ax.set_ylim(1e-5, 1)
        ax.yaxis.set_major_locator(plt.FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]))
        ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-5, 1) for j in range(2, 10)]))
        plt.plot(test_num, race_mse, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
        plt.plot(test_num, dm_mse[index], linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='purple', label="DM-KDE")
        plt.plot(test_num, pm_mse[index], linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='green', label="PM-KDE")
        plt.plot(test_num, sw_mse[index], linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='orange', label="SW-KDE")
        plt.plot(test_num, gi_mse[index], linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='blue', label="GI-KDE")
        plt.plot(test_num, mldp_kde_mse[index], linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='red', label="mLDP-KDE")
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        plt.xlabel(r'$n$', fontname='Times New Roman', fontsize=35)
        plt.ylabel('MSE')
        plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
        plt.title(rf'SYN $(\varepsilon = {e})$', fontsize=30)
        plt.show()


def draw_n_construction_time(test_num, race_ctime, gi_ctime, pm_ctime, dm_ctime, sw_ctime,
                             mldp_kde_ctime_e_1, mldp_kde_ctime_e_20, mldp_kde_ctime_e_50):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    min = 1e4 - 2e3 - 3e2
    max = 1e6 + 3e5
    ax.set_xlim(min, max)
    ax.set_ylim(1e-2, 1e6)
    ax.xaxis.set_major_locator(plt.FixedLocator([1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]))
    ax.xaxis.set_minor_locator(plt.FixedLocator([]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-2, 7) for j in range(2, 10)]))
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-2, 1e0, 1e2, 1e4, 1e6]))
    plt.plot(test_num, race_ctime, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.plot(test_num, dm_ctime, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='purple', label="DM-KDE")
    plt.plot(test_num, pm_ctime, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='green', label="PM-KDE")
    plt.plot(test_num, sw_ctime, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='orange', label="SW-KDE")
    plt.plot(test_num, gi_ctime, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='blue', label="GI-KDE")
    plt.plot(test_num, mldp_kde_ctime_e_1, linestyle='-.', marker='<', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='indianred', label="mLDP-KDE")
    plt.plot(test_num, mldp_kde_ctime_e_20, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='red', label="mLDP-KDE")
    plt.plot(test_num, mldp_kde_ctime_e_50, linestyle='--', marker='>', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='darkred', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$n$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('Construction Time (s)')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(fr'SYN', fontsize=30)
    plt.show()


def draw_n_query_time(test_num, race_qtime, gi_qtime, pm_qtime, dm_qtime, sw_qtime,
                      mldp_kde_qtime_e_1, mldp_kde_qtime_e_20, mldp_kde_qtime_e_50):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    min = 1e4 - 2e3 - 3e2
    max = 1e6 + 3e5
    ax.set_xlim(min, max)
    ax.set_ylim(1e-6, 1e2)
    ax.xaxis.set_major_locator(plt.FixedLocator([1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]))
    ax.xaxis.set_minor_locator(plt.FixedLocator([]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 3) for j in range(2, 10)]))
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-6, 1e-4, 1e-2, 1e0, 1e2]))
    plt.plot(test_num, race_qtime, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.plot(test_num, dm_qtime, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='purple', label="DM-KDE")
    plt.plot(test_num, pm_qtime, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='green', label="PM-KDE")
    plt.plot(test_num, sw_qtime, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='orange', label="SW-KDE")
    plt.plot(test_num, gi_qtime, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='blue', label="GI-KDE")
    plt.plot(test_num, mldp_kde_qtime_e_1, linestyle='-.', marker='<', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='indianred', label=r"mLDP-KDE ($\varepsilon = 1$)")
    plt.plot(test_num, mldp_kde_qtime_e_20, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='red', label=r"mLDP-KDE ($\varepsilon = 20$)")
    plt.plot(test_num, mldp_kde_qtime_e_50, linestyle='--', marker='>', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='darkred', label=r"mLDP-KDE ($\varepsilon = 50$)")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$n$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('Query Time (s)')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(fr'SYN', fontsize=30)
    plt.show()


def draw_m_MSE(test_m, race_mse, gi_mse, pm_mse, dm_mse, sw_mse, mldp_kde_mse):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    test_epsilon = [1, 20, 50]
    for index, e in enumerate(test_epsilon):
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xlim(2, 53)
        ax.set_ylim(1e-5, 1)
        ax.xaxis.set_major_locator(plt.FixedLocator([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
        ax.yaxis.set_major_locator(plt.FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]))
        ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-5, 1) for j in range(2, 10)]))
        plt.plot(test_m, race_mse, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
        plt.plot(test_m, dm_mse[index], linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='purple', label="DM-KDE")
        plt.plot(test_m, pm_mse[index], linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='green', label="PM-KDE")
        plt.plot(test_m, sw_mse[index], linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='orange', label="SW-KDE")
        plt.plot(test_m, gi_mse[index], linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='blue', label="GI-KDE")
        plt.plot(test_m, mldp_kde_mse[index], linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
                 clip_on=False, color='red', label="mLDP-KDE")
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        plt.xlabel(r'$m$', fontname='Times New Roman', fontsize=35)
        plt.ylabel('MSE')
        plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
        plt.title(rf'SYN $(\varepsilon = {e})$', fontsize=30)
        plt.show()


def draw_m_construction_time(test_m, race_ctime, gi_ctime, pm_ctime, dm_ctime, sw_ctime,
                             mldp_kde_ctime_e_1, mldp_kde_ctime_e_20, mldp_kde_ctime_e_50):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.set_yscale('log')
    ax.set_xlim(2, 53)
    ax.set_ylim(5e-1, 100)
    ax.xaxis.set_major_locator(plt.FixedLocator([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-1, 3) for j in range(2, 10)]))
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
    plt.plot(test_m, race_ctime, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.plot(test_m, dm_ctime, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='purple', label="DM-KDE")
    plt.plot(test_m, pm_ctime, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='green', label="PM-KDE")
    plt.plot(test_m, sw_ctime, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='orange', label="SW-KDE")
    plt.plot(test_m, gi_ctime, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='blue', label="GI-KDE")
    plt.plot(test_m, mldp_kde_ctime_e_1, linestyle='-.', marker='<', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='indianred', label="mLDP-KDE")
    plt.plot(test_m, mldp_kde_ctime_e_20, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='red', label="mLDP-KDE")
    plt.plot(test_m, mldp_kde_ctime_e_50, linestyle='--', marker='>', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='darkred', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$m$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('Construction Time (s)')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(fr'SYN', fontsize=30)
    plt.show()


def draw_m_query_time(test_m, race_qtime, gi_qtime, pm_qtime, dm_qtime, sw_qtime,
                      mldp_kde_qtime_e_1, mldp_kde_qtime_e_20, mldp_kde_qtime_e_50):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
    ax.set_yscale('log')
    ax.set_xlim(2, 53)
    ax.set_ylim(1e-6, 1e2)
    ax.yaxis.set_minor_locator(plt.FixedLocator([10 ** i * j for i in range(-6, 3) for j in range(2, 10)]))
    ax.yaxis.set_major_locator(plt.FixedLocator([1e-6, 1e-4, 1e-2, 1e0, 1e2]))
    plt.plot(test_m, race_qtime, linestyle='-', linewidth=2, markerfacecolor='none', clip_on=False, color='black', label="RACE")
    plt.plot(test_m, dm_qtime, linestyle='-', marker='^', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='purple', label="DM-KDE")
    plt.plot(test_m, pm_qtime, linestyle='-', marker='s', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='green', label="PM-KDE")
    plt.plot(test_m, sw_qtime, linestyle='-', marker='D', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='orange', label="SW-KDE")
    plt.plot(test_m, gi_qtime, linestyle='-', marker='x', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='blue', label="GI-KDE")
    plt.plot(test_m, mldp_kde_qtime_e_1, linestyle='-.', marker='<', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='indianred', label="mLDP-KDE")
    plt.plot(test_m, mldp_kde_qtime_e_20, linestyle='-', marker='o', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='red', label="mLDP-KDE")
    plt.plot(test_m, mldp_kde_qtime_e_50, linestyle='--', marker='>', linewidth=1, markersize=15, markerfacecolor='none', markeredgewidth=3,
             clip_on=False, color='darkred', label="mLDP-KDE")
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel(r'$m$', fontname='Times New Roman', fontsize=35)
    plt.ylabel('Query Time (s)')
    plt.grid(True, linestyle='-', alpha=0.5, color='lightgray')
    plt.title(r'SYN', fontsize=30)
    plt.show()


def draw_heatmap(query_data_embedded, acc_kde_vals, plot_value, datasets, selected_flag, e, r, method_flag):
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 30,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })
    x_min = -70
    x_max = 70
    y_min = -70
    y_max = 70
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    kde_values = griddata((query_data_embedded[:, 0], query_data_embedded[:, 1]), plot_value, (x, y), method='nearest')
    kde_values_smooth = median_filter(kde_values, size=15)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.FixedLocator([-50, 0, 50]))
    ax.yaxis.set_major_locator(plt.FixedLocator([-50, 0, 50]))
    plt.imshow(kde_values_smooth, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='coolwarm', alpha=1, vmin=np.min(acc_kde_vals),
               vmax=np.max(acc_kde_vals))
    plt.colorbar(label='KDE Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    if (plot_value == acc_kde_vals).all():
        plt.title(f'{datasets[selected_flag]}')
        plt.suptitle('Exact', y=-0.1, x=0.46)
    elif method_flag == 'mldp-kde':
        plt.title(f'{datasets[selected_flag]}($\\varepsilon$ = {e}, $r = {r}$)')
        plt.suptitle('mLDP-KDE', y=-0.1, x=0.46)
    elif method_flag == 'race':
        plt.title(f'{datasets[selected_flag]}')
        plt.suptitle('RACE', y=-0.1, x=0.46)
    elif method_flag == 'gi':
        plt.title(f'{datasets[selected_flag]}($\\varepsilon$ = {e})')
        plt.suptitle('GI', y=-0.1, x=0.46)
    plt.show()
