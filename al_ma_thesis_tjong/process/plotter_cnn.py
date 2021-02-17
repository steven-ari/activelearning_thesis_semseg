import os
import statistics
from os.path import dirname as dr, abspath
import math
import csv
from typing import List
import scipy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import itertools


def csv_train_reader(csv_path, data_name, graph_name, graph_title):

    title_list = []

    random_acc_list = []
    ve_acc_list = []
    ce_acc_list = []
    data_list = []

    random_acc_list_temp = []
    ve_acc_list_temp = []
    ce_acc_list_temp = []
    data_list_temp = []

    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        is_first_title = True
        for row in csv_reader:
            if is_first_title:  # first title
                title_list.append(row[0])
                is_first_title = False
            elif len(row) == 1:  # title
                if 'Random' in title_list[-1]:
                    random_acc_list.append(np.array(random_acc_list_temp))
                    data_list.append(np.array(data_list_temp))
                elif 'CE: ' in title_list[-1]:
                    ce_acc_list.append(np.array(ce_acc_list_temp))
                    data_list.append(np.array(data_list_temp))
                elif 'VE: ' in title_list[-1]:
                    ve_acc_list.append(np.array(ve_acc_list_temp))
                    data_list.append(np.array(data_list_temp))

                random_acc_list_temp = []
                ce_acc_list_temp = []
                ve_acc_list_temp = []
                data_list_temp = []

                title_list.append(row[0])
            else:  # data
                if 'Random' in title_list[-1]:
                    random_acc_list_temp.append(float(row[1]))
                    data_list_temp.append(int(row[-1]))
                elif 'CE: ' in title_list[-1]:
                    ce_acc_list_temp.append(float(row[1]))
                    data_list_temp.append(int(row[-1]))
                elif 'VE: ' in title_list[-1]:
                    ve_acc_list_temp.append(float(row[1]))
                    data_list_temp.append(int(row[-1]))

        if 'Random' in title_list[-1]:
            random_acc_list.append(np.array(random_acc_list_temp))
        elif 'CE: ' in title_list[-1]:
            ce_acc_list.append(np.array(ce_acc_list_temp))
        elif 'VE: ' in title_list[-1]:
            ve_acc_list.append(np.array(ve_acc_list_temp))

    csv_data = {
        "data_name": data_name,
        "graph_name": graph_name,
        "graph_title": graph_title,
        "titles": title_list,
        "random": random_acc_list,
        "consensus": ce_acc_list,
        "vote": ve_acc_list,
        "data_size": data_list[0],
    }
    return csv_data


def shaded_std_line(idx_plot, csv_dict, figure, label, color):
    n_plots = 10
    std = csv_dict["committee_vote_std_list"][idx_plot]
    # indices = [0] + list(range(int(std.shape[0] / n_plots) - 1, std.shape[0], int(std.shape[0] / n_plots)))
    indices = list(range(std.shape[0]))
    batch_size = csv_dict["batch_size_list"][idx_plot][indices]
    committee_vote = csv_dict["committee_vote_list"][idx_plot][indices]
    std = csv_dict["committee_vote_std_list"][idx_plot][indices]
    plt.fill_between(batch_size, committee_vote - std, committee_vote + std, alpha=0.7, color=color, figure=figure)
    plt.plot(batch_size, committee_vote, color=color, label=label, figure=figure)


def plot_shaded_std(data, color_pallete, my_dpi, test_cnn_path):
    plt.interactive(True)
    name = 'entire_data'
    max_val = True
    legend_loc = 7
    x_fontsize = 10
    plot_max = False

    figure = plt.figure(figsize=(1200 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    plots = ['consensus', 'vote', 'random']
    for i_plots in range(len(plots)):
        # get data to plot
        mean = np.mean(np.array(data[plots[i_plots]]), axis=0)
        std = np.var(np.array(data[plots[i_plots]]), axis=0)**0.5
        x_data = data['data_size']
        color = color_pallete[i_plots]
        label = plots[i_plots].capitalize()

        # plot it
        plt.fill_between(x_data, mean-std, mean+std, alpha=0.7, color=color, figure=figure)
        plt.plot(x_data, mean, color=color, figure=figure, label=label)

    plt.title('Test on: ' + data['graph_name'].replace('_', ' '))
    plt.grid(which='both', axis='both')
    plt.xlabel('Number of training samples')
    plt.legend(loc=legend_loc)
    plt.ylabel('Test data acc [%]')
    plt.xticks(ticks=x_data, fontsize=x_fontsize)
    plt.tight_layout()

    # save figure
    dir_number = 1
    result_dir = os.path.join(test_cnn_path, 'figure_' + '{:03d}'.format(dir_number))
    while os.path.isfile(result_dir):
        dir_number = dir_number + 1
        result_dir = os.path.join(test_cnn_path, 'figure_' + '{:03d}'.format(dir_number))
    file_name = result_dir + '.png'
    plt.savefig(file_name, format='png', dpi=100)


def plot_shaded_std_fig(figure, data, color_pallete, my_dpi, test_cnn_path, alpha, title):

    plots = ['consensus', 'vote', 'random']
    plot_labels = ['QBC: Consensus', 'QBC: Vote', 'without AL']
    xlabel_text = 'Number of training samples'
    # for i_plots in range(len(plots)):
    for i_plots in range(3):
        # get data to plot
        mean = np.mean(np.array(data[plots[i_plots]]), axis=0)
        sem = np.std(np.array(data[plots[i_plots]]), axis=0)/(np.array(data[plots[i_plots]]).shape[0]**0.5)
        # sem = np.std(np.array(data[plots[i_plots]]), axis=0)
        x_data = data['data_size']
        if title == 'Incremental Training':
            x_data = np.array([np.sum(x_data[0:x_data.tolist().index(x)+1]) for x in x_data])
            xlabel_text = 'Number of learned samples'
        color = color_pallete[i_plots]
        label = plot_labels[i_plots]
        # label = data['graph_name'] + ':  ' + plots[i_plots].capitalize()

        # plot it
        plt.fill_between(x_data, mean-sem, mean+sem, alpha=0.25, color=color, figure=figure)
        plt.plot(x_data, mean, color=color, figure=figure, label=label, alpha=alpha+0.25)

    plt.title('Test on: ' + title.replace('_', ' '))
    plt.grid(which='both', axis='both')
    plt.xlabel(xlabel_text)
    # plt.legend(loc=4)
    # plt.ylabel('Test data acc [%]')
    # plt.xticks(ticks=x_data, fontsize=x_fontsize)
    plt.tight_layout()


def main():

    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    my_dpi = 150

    plt.interactive(True)
    figure = plt.figure(figsize=(2400 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    graph_types = ['only_new_data', 'always_from_scratch', 'incremental_with_step']
    # graph_types = ['always_from_scratch', 'always_from_scratch_20000']
    graph_title = ['Naïve approach', 'Sample Transfer', 'Incremental Training']
    # graph_title = ['Sample Transfer (AL Batch: 3000)', 'Sample Transfer (AL Batch: 1000)', 'Incremental Training']
    alpha = [0.75, 0.75, 0.75]
    data_all = []
    axes = []
    for i_graphs in range(len(graph_types)):
        pos = 13*10+(i_graphs+1)
        axes.append(plt.subplot(pos))
        graph_type = graph_types[i_graphs]
        test_cnn_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'unreduced_mnist', graph_type, 'cnn_qbc')
        csv_test_cnn_path = os.path.join(test_cnn_path, 'test.csv')
        print(csv_test_cnn_path)
        data = csv_train_reader(csv_test_cnn_path, 'unreduced_MNIST', graph_type, graph_title)
        data_all.append(data)
        plot_shaded_std_fig(figure, data, color_palette, my_dpi, test_cnn_path, alpha[i_graphs], graph_title[i_graphs])

    plt.setp(axes[1].get_yticklabels(), visible=True)
    # plt.setp(axes[2].get_yticklabels(), visible=True)
    axes[0].set_ylabel("Test data acc [%]")
    axes[1].set_ylabel("Test data acc [%]")
    # axes[0].set_ylim((0.8, 0.980))
    # axes[1].set_ylim((0.8, 0.980))
    # plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=50)
    # axes[2].set_ylabel("Test data acc [%]")
    plt.tight_layout()

    # save figure
    dir_number = 1
    result_dir = os.path.join(test_cnn_path, 'figure_' + '{:03d}'.format(dir_number))
    while os.path.isfile(result_dir):
        dir_number = dir_number + 1
        result_dir = os.path.join(test_cnn_path, 'figure_' + '{:03d}'.format(dir_number))
    file_name = result_dir + '.png'
    plt.savefig(file_name, format='png', dpi=300)


if __name__ == '__main__':
    main()



