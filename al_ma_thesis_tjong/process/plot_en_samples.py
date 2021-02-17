import os
import statistics
from os.path import dirname as dr, abspath
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.stats as stats


def main():

    # get entropy data
    en_val_ce_20 = np.load('C:\\Users\\steve\\Desktop\\projects\\al_kitti\\results\\random\\consensus_ranker_001_dr20\\entropy.npy')
    en_val_ce_90 = np.load('C:\\Users\\steve\\Desktop\\projects\\al_kitti\\results\\random\\consensus_ranker_002_dr90\\entropy.npy')
    en_val_ve_20 = np.load('C:\\Users\\steve\\Desktop\\projects\\al_kitti\\results\\random\\ve_ranker_001_dr20\\entropy.npy')
    en_val_ve_90 = np.load('C:\\Users\\steve\\Desktop\\projects\\al_kitti\\results\\random\\ve_ranker_002_dr90\\entropy.npy')

    # prepare figure
    plt.interactive(True)
    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 150
    figure, ax_en = plt.subplots(figsize=(1400 / my_dpi, 500 / my_dpi), dpi=my_dpi)

    # plot it
    plt.scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20, color=color_palette[0], label='Consensus, dropout rate: 20%', alpha=0.3)
    plt.scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90, color=color_palette[1], label='Consensus, dropout rate: 90%', alpha=0.3)
    plt.scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20, color=color_palette[2], label='Vote, dropout rate: 20%', alpha=0.3)
    plt.scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90, color=color_palette[3], label='Vote, dropout rate: 90%', alpha=0.3)

    # make plot pretty
    plt.title('Entropy Values of Samples in Cityscapes')
    plt.xlabel('Sample Index')
    plt.ylabel('Average Entropy')
    plt.tight_layout()
    ax_en.legend(fancybox=True, framealpha=0.1, loc=2)
    ax_en.grid(which='both', axis='both')

    a = 1


def calculate_trend(en_data):

    # solve least squares for Ax = B
    A = np.stack((np.arange(0, len(en_data)), np.ones_like(en_data))).T
    B = en_data
    param = np.matmul(np.linalg.pinv(A), B)

    # calculate points for trend line
    x1, y1 = 1, 1*param[0] + param[1]
    x2, y2 = len(en_data), len(en_data)*param[0] + param[1]

    # solve least squares for Ax = B
    A = np.stack((np.arange(0, len(en_data))/(len(en_data)/2)-1, np.ones_like(en_data))).T
    B = en_data
    param_norm = np.matmul(np.linalg.pinv(A), B)

    return param, param_norm, [x1, x2, y1, y2]


def compare():

    # get entropy data
    en_val_ce_20 = np.load('C:\\Users\\steve\\Desktop\\projects_software\\al_kitti\\results\\random\\consensus_ranker_001_dr20\\entropy.npy')
    en_val_ce_90 = np.load('C:\\Users\\steve\\Desktop\\projects_software\\al_kitti\\results\\random\\consensus_ranker_002_dr90\\entropy.npy')
    en_val_ve_20 = np.load('C:\\Users\\steve\\Desktop\\projects_software\\al_kitti\\results\\random\\ve_ranker_001_dr20\\entropy.npy')
    en_val_ve_90 = np.load('C:\\Users\\steve\\Desktop\\projects_software\\al_kitti\\results\\random\\ve_ranker_002_dr90\\entropy.npy')

    # prepare figure
    plt.interactive(True)
    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 180
    figure, ax_en_all = plt.subplots(2, 1, figsize=(7, 4.5), dpi=my_dpi)
    # figure.suptitle('Comparison of Four Entropy Methods on each Cityscape Sample')

    # plot with consensus 20
    sort_ce_20 = np.argsort(en_val_ce_20)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90[sort_ce_20], color=color_palette[1], label='Consensus, dropout rate: 90%', alpha=0.3)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20[sort_ce_20], color=color_palette[2], label='Vote, dropout rate: 20%', alpha=0.3)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90[sort_ce_20], color=color_palette[3], label='Vote, dropout rate: 90%', alpha=0.3)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20[sort_ce_20], color=color_palette[0],
                         label='Consensus, dropout rate: 20%', alpha=0.3)

    # calculate trend
    param, param_norm, points = calculate_trend(en_val_ce_90[sort_ce_20])
    # ax_en_all[0].plot((points[0], points[1]), (points[2], points[3]), 'k--', label='Trendline Consensus 90%')
    print(param)
    print(param_norm)

    # make plot pretty
    ax_en_all[0].legend(fancybox=True, framealpha=0.7, loc=2)
    handles, labels = ax_en_all[0].get_legend_handles_labels()
    order = [4, 1, 2, 3, 0]
    # ax_en_all[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], fancybox=True, framealpha=0.7, loc=2)
    ax_en_all[0].set_title('Sorted Based on Consensus 20%')
    ax_en_all[0].set_ylabel('Avg Image Entropy []')
    ax_en_all[0].grid(which='both', axis='both')
    ax_en_all[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # plot with consensus 90
    sort_ce_90 = np.argsort(en_val_ce_90)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20[sort_ce_90], color=color_palette[0], alpha=0.3)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20[sort_ce_90], color=color_palette[2], alpha=0.3)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90[sort_ce_90], color=color_palette[3], alpha=0.3)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90[sort_ce_90], color=color_palette[1], alpha=0.3)

    # calculate trend
    param, param_norm, points = calculate_trend(en_val_ce_20[sort_ce_90])
    # ax_en_all[1].plot((points[0], points[1]), (points[2], points[3]), 'k--', label='Trendline Consensus 20%')
    print(param)
    print(param_norm)

    # make plot pretty
    ax_en_all[1].legend(fancybox=True, framealpha=0.7, loc=2)
    ax_en_all[1].set_title('Sorted Based on Consensus 90%')
    ax_en_all[1].set_ylabel('Avg Image Entropy []')
    ax_en_all[1].set_xlabel('Sorted Sample Index')
    ax_en_all[1].grid(which='both', axis='both')

    '''# plot with vote 20
    sort_ve_20 = np.argsort(en_val_ve_20)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20[sort_ve_20], color=color_palette[0],
                         label='Consensus, dropout rate: 20%', alpha=0.3)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90[sort_ve_20], color=color_palette[1],
                         label='Consensus, dropout rate: 90%', alpha=0.3)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90[sort_ve_20], color=color_palette[3],
                         label='Vote, dropout rate: 90%', alpha=0.3)
    ax_en_all[0].scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20[sort_ve_20], color=color_palette[2],
                         label='Vote, dropout rate: 20%', alpha=0.3)

    # make plot pretty
    ax_en_all[0].legend(fancybox=True, framealpha=0.7, loc=2)
    ax_en_all[0].set_title('Sorted Based on Vote 20%')
    ax_en_all[0].set_ylabel('Avg Image Entropy []')
    ax_en_all[0].grid(which='both', axis='both')
    ax_en_all[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax_en_all[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_en_all[0].set_ylim(0, 0.28)

    # plot with vote 90
    sort_ve_90 = np.argsort(en_val_ve_90)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20[sort_ve_90], color=color_palette[0],
                         label='Consensus, dropout rate: 20%', alpha=0.3)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90[sort_ve_90], color=color_palette[1],
                         label='Consensus, dropout rate: 90%', alpha=0.3)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20[sort_ve_90], color=color_palette[2],
                         label='Vote, dropout rate: 20%', alpha=0.3)
    ax_en_all[1].scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90[sort_ve_90], color=color_palette[3],
                         label='Vote, dropout rate: 90%', alpha=0.3)

    # make plot pretty
    ax_en_all[1].set_title('Sorted Based on Vote 90%')
    ax_en_all[1].set_ylabel('Avg Image Entropy []')
    ax_en_all[1].set_xlabel('Sorted Sample Index')
    ax_en_all[1].grid(which='both', axis='both')
    ax_en_all[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_en_all[1].set_ylim(0, 0.28)'''

    '''# plot with vote 20
    sort_ve_20 = np.argsort(en_val_ve_20)
    ax_en_all[2].scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20[sort_ve_20], color=color_palette[0], label='Consensus, dropout rate: 20%', alpha=0.3)
    ax_en_all[2].scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90[sort_ve_20], color=color_palette[1], label='Consensus, dropout rate: 90%', alpha=0.3)
    ax_en_all[2].scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90[sort_ve_20], color=color_palette[3], label='Vote, dropout rate: 90%', alpha=0.3)
    ax_en_all[2].scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20[sort_ve_20], color=color_palette[2], label='Vote, dropout rate: 20%', alpha=0.3)

    # make plot pretty
    ax_en_all[2].legend(fancybox=True, framealpha=0.7, loc=2)
    # ax_en_all[2].set_title('Sorted Based on Vote 20%')
    ax_en_all[2].set_ylabel('Avg Image Entropy []')
    ax_en_all[2].grid(which='both', axis='both')
    ax_en_all[2].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax_en_all[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_en_all[2].set_ylim(0, 0.28)

    # plot with vote 90
    sort_ve_90 = np.argsort(en_val_ve_90)
    ax_en_all[3].scatter(np.arange(0, len(en_val_ce_20)), en_val_ce_20[sort_ve_90], color=color_palette[0], label='Consensus, dropout rate: 20%', alpha=0.3)
    ax_en_all[3].scatter(np.arange(0, len(en_val_ce_90)), en_val_ce_90[sort_ve_90], color=color_palette[1], label='Consensus, dropout rate: 90%', alpha=0.3)
    ax_en_all[3].scatter(np.arange(0, len(en_val_ve_20)), en_val_ve_20[sort_ve_90], color=color_palette[2], label='Vote, dropout rate: 20%', alpha=0.3)
    ax_en_all[3].scatter(np.arange(0, len(en_val_ve_90)), en_val_ve_90[sort_ve_90], color=color_palette[3], label='Vote, dropout rate: 90%', alpha=0.3)

    # make plot pretty
    ax_en_all[3].set_title('Sorted Based on Vote 90%')
    ax_en_all[3].set_ylabel('Avg Image Entropy []')
    ax_en_all[3].set_xlabel('Sorted Sample Index')
    ax_en_all[3].grid(which='both', axis='both')
    ax_en_all[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_en_all[3].set_ylim(0, 0.28)'''

    plt.tight_layout()

    a = 1


if __name__ == '__main__':
    compare()