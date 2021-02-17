import os
import statistics
from os.path import dirname as dr, abspath
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import itertools


def csv_scanner(csv_path, data_name):
    epoch_list = []
    n_data_list = []
    iou_list = []

    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        is_first_title = True
        for row in csv_reader:
            if is_first_title:  # first title
                title = row[0]
                is_first_title = False
            else:
                epoch_list.append(int(row[0]))
                n_data_list.append(int(row[-1]))
                iou_list.append(float(row[1]))
    data = {
        "data_name": data_name,
        "epoch": epoch_list,
        "n_data": n_data_list,
        "iou": iou_list,
    }

    return data


def plot_shaded_std_amount(data_all, color_pallete, my_dpi, title_info):

    plt.interactive(True)
    legend_loc = 7
    x_fontsize = 10

    figure, ax_iou = plt.subplots(figsize=(1300 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    plt.xlabel('epochs')
    plt.ylabel('Test data IOU')
    ax_bar = ax_iou.twinx()
    label_list = ["QBC: Consensus,", "QBC: Vote,          ", "Without AL,        "]
    title = 'AL vs Random Data Increment, avg IoU differrence : '
    end_val = []

    for i_data in range(len(data_all)):
        iou_list = []
        data_current = data_all[i_data]
        x_data = data_current[0]['epoch']
        for i_plot in range(len(data_current)):
            # get data to plot
            iou_list.append(np.array(data_current[i_plot]['iou']))

        iou = np.array(iou_list)
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)/(iou.shape[0]**0.5)

        end_val.append(iou_mean[-1])

        # plot it
        ax_iou.fill_between(x_data, iou_mean - iou_std, iou_mean + iou_std, alpha=0.2,
                            color=color_pallete[i_data], figure=figure)
        ax_iou.plot(x_data, iou_mean, color=color_pallete[i_data], figure=figure,
                    label=label_list[i_data] + " end IoU:" + "{:1.4f}".format(iou_mean[-1]))

    # plot bar for amount of data:
    ax_bar.bar(data_current[i_plot]['epoch'], data_current[i_plot]['n_data'], width=1,
               color=color_pallete[4],
               label=label_list[i_data] + " end IoU:" + "{:1.4f}".format(iou_mean[-1]),
               figure=figure, alpha=0.2)

    plt.yticks(data_current[-1]['n_data'])

    plt.title(title + "{:1.4f}".format(
        np.mean(np.array([end_val[0], end_val[1]])) - end_val[-1]) + " (" + title_info +" Data)")
    plt.xticks(ticks=x_data, fontsize=x_fontsize)
    plt.ylabel('Num of samples')
    plt.tight_layout()
    ax_iou.set_zorder(ax_bar.get_zorder() + 1)
    ax_iou.patch.set_visible(False)
    ax_iou.legend(loc=legend_loc)
    # ax_bar.grid(which='both', axis='both')
    ax_iou.grid(which='both', axis='both')


def plot_shaded_std(data_all, color_pallete, my_dpi, title_info):

    plt.interactive(True)
    name = 'entire_data'
    max_val = True
    legend_loc = 7
    x_fontsize = 10
    plot_max = False

    figure, ax_iou = plt.subplots(figsize=(1200 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    plt.xlabel('epochs')
    plt.ylabel('Test data IOU')
    label_list = ["AL: Consensus", "          AL: Vote", "          Random"]
    title = 'AL vs Random Data Increment, IoU differrence : '
    end_val = []

    for i_data in range(len(data_all)):
        iou_list = []
        data_current = data_all[i_data]
        x_data = data_current[0]['epoch']
        for i_plot in range(len(data_current)):
            # get data to plot
            iou_list.append(np.array(data_current[i_plot]['iou']))

        iou = np.array(iou_list)
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)/(iou.shape[0]**0.5)

        end_val.append(iou_mean[-1])

        # plot it
        ax_iou.fill_between(x_data, iou_mean - iou_std, iou_mean + iou_std, alpha=0.2,
                            color=color_pallete[i_data], figure=figure)
        ax_iou.plot(x_data, iou_mean, color=color_pallete[i_data], figure=figure,
                    label=label_list[i_data] + ", end IoU:" + "{:1.4f}".format(iou_mean[-1]))

    plt.title(title + "{:1.4f}".format(
        np.mean(np.array([end_val[0], end_val[1]])) - end_val[-1]) + " (" + title_info +" Data)")
    plt.xticks(ticks=x_data, fontsize=x_fontsize)
    plt.tight_layout()
    ax_iou.legend(loc=legend_loc)
    ax_iou.grid(which='both', axis='both')


def mini_20():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 120

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_consensus_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_consensus_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_consensus_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_consensus_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_consensus_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_vote_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_vote_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_vote_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_vote_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_getter_from_vote_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_20_mini_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
    """
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '20%')

    a = 1

    return [data_consensus, data_vote, data_random]


def mini_30():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 120

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_consensus_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_consensus_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_consensus_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_consensus_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_consensus_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_vote_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_vote_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_vote_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_vote_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_getter_from_vote_90_004_002', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_30_mini_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
    """
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '30%')

    a = 1

    return [data_consensus, data_vote, data_random]


def mini_40():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 120

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_consensus_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_consensus_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_consensus_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_consensus_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_consensus_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_vote_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_vote_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_vote_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_vote_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_getter_from_vote_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_40_mini_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
    """
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '40%')

    a = 1

    return [data_consensus, data_vote, data_random]


def mini_70():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 120

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_consensus_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_consensus_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_consensus_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_consensus_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_consensus_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_vote_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_vote_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_vote_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_vote_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_getter_from_vote_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'mini', 'bulk_70_mini_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
    """
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '70%')

    a = 1

    return [data_consensus, data_vote, data_random]


def mini():

    data_20_mini = bulk_to_all(mini_20())
    data_30_mini = bulk_to_all(mini_30())
    data_40_mini = bulk_to_all(mini_40())
    data_70_mini = bulk_to_all(mini_70())

    plt.interactive(True)
    my_dpi = 100

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')

    figure, ax_iou = plt.subplots(figsize=(500 / my_dpi, 500 / my_dpi), dpi=my_dpi)
    data = [data_20_mini, data_30_mini, data_40_mini, data_70_mini]
    x_data = np.array([20, 30, 40, 70])

    i_data = 0

    # plot consensus
    mean = np.mean(data[i_data]['consensus'])
    std = np.std(data[i_data]['consensus'])
    plt.errorbar(x_data[i_data]-1, mean, yerr=std, color=color_palette[0], label='AL1: Consensus', alpha=0.4, capsize=5, capthick=2)
    plt.scatter(x_data[i_data]-1, mean, color=color_palette[0], alpha=1)
    consensus_mean = mean

    # plot vote
    mean = np.mean(data[i_data]['vote'])
    std = np.std(data[i_data]['vote'])
    plt.errorbar(x_data[i_data]+1, mean, yerr=std, color=color_palette[1], label='AL2: Vote', alpha=0.4, capsize=5, capthick=2)
    plt.scatter(x_data[i_data]+1, mean, color=color_palette[1], alpha=1)
    vote_mean = mean

    # plot random
    mean = np.mean(data[i_data]['random'])
    std = np.std(data[i_data]['random'])
    plt.errorbar(x_data[i_data], mean, yerr=std, color=color_palette[2], label='random', alpha=0.4, capsize=5, capthick=2)
    plt.scatter(x_data[i_data], mean, color=color_palette[2], alpha=1)
    random_mean = mean
    diff = np.absolute(np.mean([vote_mean, consensus_mean])-random_mean)
    plt.text(x_data[i_data]+1, random_mean, "Avg diff: \n{:1.4f} IoU".format(diff), horizontalalignment='left')

    for i_data in range(1, len(data)):

        # plot consensus
        mean = np.mean(data[i_data]['consensus'])
        std = np.std(data[i_data]['consensus'])
        plt.errorbar(x_data[i_data]-1, mean, yerr=std, color=color_palette[0], alpha=0.4, capsize=5, capthick=2)
        plt.scatter(x_data[i_data]-1, mean, color=color_palette[0], alpha=1)
        consensus_mean = mean

        # plot vote
        mean = np.mean(data[i_data]['vote'])
        std = np.std(data[i_data]['vote'])
        plt.errorbar(x_data[i_data]+1, mean, yerr=std, color=color_palette[1], alpha=0.4, capsize=5, capthick=2)
        plt.scatter(x_data[i_data]+1, mean, color=color_palette[1], alpha=1)
        vote_mean = mean

        # plot random
        mean = np.mean(data[i_data]['random'])
        std = np.std(data[i_data]['random'])
        plt.errorbar(x_data[i_data], mean, yerr=std, color=color_palette[2], alpha=0.4, capsize=5, capthick=2)
        plt.scatter(x_data[i_data], mean, color=color_palette[2], alpha=1)
        random_mean = mean

        diff = np.absolute(np.mean([vote_mean, consensus_mean])-random_mean)
        if i_data != (len(data)-1):
            alg = 'left'
            x_text = x_data[i_data]+1
        else:
            alg = 'right'
            x_text = x_data[i_data]-1
        plt.text(x_text, random_mean, "Avg diff: \n{:1.4f} IoU".format(diff), horizontalalignment=alg)

    plt.ylabel('Test data IOU')
    plt.xticks(x_data)
    plt.xlabel('% of samples in Cityscapes')
    ax_iou.grid(which='both', axis='y')
    ax_iou.legend(loc=4)
    # ax_iou.legend(loc=4, bbox_to_anchor=(1, 1.05))
    # figure.suptitle("Cityscapes (AL vs Standard)")
    figure.suptitle("Cityscapes Standard Training")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    a = 1


def bulk_20():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 100

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_20_from_90_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_20_from_90_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_20_from_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_20_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_20_from_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_20_from_90_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_20_from_90_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_20_from_90_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_20_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_20_from_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_20_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_20_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_20_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_20_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_20_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
"""
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '20%')

    a = 1

    return [data_consensus, data_vote, data_random]


def bulk_30():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 100

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_30_from_90_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_30_from_90_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_30_from_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_30_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_30_from_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_30_from_90_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_30_from_90_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_30_from_90_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_30_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_30_from_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_30_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_30_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_30_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_30_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_30_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
"""
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '30%')

    a = 1

    return [data_consensus, data_vote, data_random]


def bulk_40():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 100

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_40_from_90_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_40_from_90_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_40_from_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_40_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_40_from_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_40_from_90_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_40_from_90_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_40_from_90_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_40_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_40_from_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_40_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_40_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_40_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_40_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_40_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
"""
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '40%')

    a = 1

    return [data_consensus, data_vote, data_random]


def bulk_70():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 100

    # Pretrained or increment
    csv_paths_consensus = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_70_from_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_70_from_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_70_from_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_70_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_bulk_70_from_90_005_001', 'test.csv'),
    ]
    csv_paths_ve = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_70_from_90_001_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_70_from_90_002_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_70_from_90_003_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_70_from_90_004_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_bulk_70_from_90_005_001', 'test.csv'),
    ]
    csv_paths_random = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_70_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_70_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_70_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_70_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_bulk_70_005', 'test.csv'),
        ]

    """titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))
"""
    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(len(csv_paths_consensus)):
        data_consensus.append(csv_scanner(csv_paths_consensus[i_data], "-"))

    for i_data in range(len(csv_paths_ve)):
        data_vote.append(csv_scanner(csv_paths_ve[i_data], "-"))

    for i_data in range(len(csv_paths_random)):
        data_random.append(csv_scanner(csv_paths_random[i_data], "-"))

    plot_shaded_std([data_consensus, data_vote, data_random], color_palette, my_dpi, '70%')

    a = 1

    return [data_consensus, data_vote, data_random]


def pretrained_50():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 120

    # Pretrained or increment
    csv_paths = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_005', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_005', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_50_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_50_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_50_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_50_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_50_005', 'test.csv'),
        ]

    titles = ['consensus_5_DR90_001',
              'consensus_5_DR90_002',
              'consensus_5_DR90_003',
              'consensus_5_DR90_004',
              'consensus_5_DR90_005',
              'vote_5_DR90_001',
              'vote_5_DR90_002',
              'vote_5_DR90_003',
              'vote_5_DR90_004',
              'vote_5_DR90_005',
              'random_5_DR90_001',
              'random_5_DR90_002',
              'random_5_DR90_003',
              'random_5_DR90_004',
              'random_5_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(5):
        data_consensus.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    for i_data in range(5, 10):
        data_vote.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    for i_data in range(10, 15):
        data_random.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    plot_shaded_std_amount([data_consensus, data_vote, data_random], color_palette, my_dpi, "50%")

    a = 1


def pretrained_70():

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')
    my_dpi = 120

    # Pretrained or increment
    csv_paths = [
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_70_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_70_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_70_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_70_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test', 'consensus_90_5_70_005', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_70_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_70_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_70_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_70_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test', 'vote_90_5_70_005', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_70_001', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_70_002', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_70_003', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_70_004', 'test.csv'),
        os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'random', 'random_5_70_005', 'test.csv'),
        ]

    titles = ['consensus_5_70_DR90_001',
              'consensus_5_70_DR90_002',
              'consensus_5_70_DR90_003',
              'consensus_5_70_DR90_004',
              'consensus_5_70_DR90_005',
              'vote_5_70_DR90_001',
              'vote_5_70_DR90_002',
              'vote_5_70_DR90_003',
              'vote_5_70_DR90_004',
              'vote_5_70_DR90_005',
              'random_5_70_DR90_001',
              'random_5_70_DR90_002',
              'random_5_70_DR90_003',
              'random_5_70_DR90_004',
              'random_5_70_DR90_005',
              ]
    fig_title = "Pretrained setting, 50 epochs, increase every 5 epoch until 50% Cityscapes"

    data_all = []

    for i_data in range(len(csv_paths)):
        data_all.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    # plot_single(data_all, color_palette, my_dpi, fig_title)

    data_consensus = []
    data_vote = []
    data_random = []

    for i_data in range(5):
        data_consensus.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    for i_data in range(5, 10):
        data_vote.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    for i_data in range(10, 15):
        data_random.append(csv_scanner(csv_paths[i_data], titles[i_data]))

    plot_shaded_std_amount([data_consensus, data_vote, data_random], color_palette, my_dpi, "70%")

    a = 1


def bulk_to_all(data_all):

    data_consensus = data_all[0]
    data_vote = data_all[1]
    data_random = data_all[2]

    iou_end_consensus = []
    for i_data in range(len(data_consensus)):
        iou_end_consensus.append(data_consensus[i_data]['iou'][-1])

    iou_end_vote = []
    for i_data in range(len(data_vote)):
        iou_end_vote.append(data_vote[i_data]['iou'][-1])

    iou_end_random = []
    for i_data in range(len(data_random)):
        iou_end_random.append(data_random[i_data]['iou'][-1])

    dict_end = {
        'random': np.array(iou_end_random),
        'consensus': np.array(iou_end_consensus),
        'vote': np.array(iou_end_vote),
    }

    return dict_end


def bulk():

    data_20 = bulk_to_all(bulk_20())
    data_30 = bulk_to_all(bulk_30())
    data_40 = bulk_to_all(bulk_40())
    data_70 = bulk_to_all(bulk_70())

    plt.interactive(True)
    name = 'entire_data'
    max_val = True
    legend_loc = 7
    x_fontsize = 10
    plot_max = False
    my_dpi = 100

    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                     'b', 'g', 'r', 'c', 'm', 'y')

    figure, ax_iou = plt.subplots(figsize=(500 / my_dpi, 500 / my_dpi), dpi=my_dpi)
    data = [data_20, data_30, data_40, data_70]
    x_data = np.array([20, 30, 40, 70])
    i_data = 0

    consensus_mean = []
    consensus_std = []
    vote_mean = []
    vote_std = []
    random_mean = []
    random_std = []

    # plot consensus
    consensus_mean.append(np.mean(data[i_data]['consensus']))
    consensus_std.append(np.std(data[i_data]['consensus']))

    # plot vote
    vote_mean.append(np.mean(data[i_data]['vote']))
    vote_std.append(np.std(data[i_data]['vote']))

    # plot random
    random_mean.append(np.mean(data[i_data]['random']))
    random_std.append(np.std(data[i_data]['random']))
    diff = np.absolute(np.mean([vote_mean[i_data], consensus_mean[i_data]])-random_mean[i_data])
    plt.text(x_data[i_data]+1, random_mean[i_data], "Avg diff: \n{:1.4f} IoU".format(diff), horizontalalignment='left')

    for i_data in range(1, len(data)):

        # plot consensus
        consensus_mean.append(np.mean(data[i_data]['consensus']))
        consensus_std.append(np.std(data[i_data]['consensus']))

        # plot vote
        vote_mean.append(np.mean(data[i_data]['vote']))
        vote_std.append(np.std(data[i_data]['vote']))

        # plot random
        random_mean.append(np.mean(data[i_data]['random']))
        random_std.append(np.std(data[i_data]['random']))

        diff = np.absolute(np.mean([vote_mean[i_data], consensus_mean[i_data]]) - random_mean[i_data])
        if i_data != (len(data)-1):
            alg = 'left'
            x_text = x_data[i_data]+1
        else:
            alg = 'right'
            x_text = x_data[i_data]-1
        plt.text(x_text, random_mean[i_data], "Avg diff: \n{:1.4f} IoU".format(diff),
                horizontalalignment='left')

    # plot shaded std
    plt.fill_between(x_data, np.array(consensus_mean)-np.array(consensus_std),
                     np.array(consensus_mean)+np.array(consensus_std), alpha=0.19, color=color_palette[0])
    plt.fill_between(x_data, np.array(vote_mean)-np.array(vote_std),
                     np.array(vote_mean)+np.array(vote_std), alpha=0.19, color=color_palette[1])
    plt.fill_between(x_data, np.array(random_mean)-np.array(random_std),
                     np.array(random_mean)+np.array(random_std), alpha=0.19, color=color_palette[2])
    plt.plot(x_data, np.array(consensus_mean), color=color_palette[0], label='AL variant 1: Consensus')
    plt.plot(x_data, np.array(vote_mean), color=color_palette[1], label='AL variant 2: Vote')
    plt.plot(x_data, np.array(random_mean), color=color_palette[2], label='without AL')
    left, right = plt.xlim()
    plt.hlines(0.703, 0, 80, color='k', linestyles='dashed')
    plt.text(20, 0.696, '100% Cityscapes')
    plt.xlim((left, right))

    plt.ylabel('Test data IOU')
    plt.xticks(x_data)
    plt.xlabel('% of samples in Cityscapes')
    ax_iou.grid(which='both', axis='y')
    ax_iou.legend(loc=4)
    # ax_iou.legend(loc=4, bbox_to_anchor=(1, 1.05))
    figure.suptitle("Cityscapes (AL vs Standard)")
    # figure.suptitle("Cityscapes Standard Training")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    a = 1


if __name__ == '__main__':
    bulk()
