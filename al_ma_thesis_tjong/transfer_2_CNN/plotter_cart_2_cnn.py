import os
import statistics
from os.path import dirname as dr, abspath
import math
import csv
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import itertools


def csv_train_reader(csv_path, data_name):

    title_list = []

    random_acc_list = []
    data_list = []
    random_acc_list_temp = []
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
                random_acc_list.append(np.array(random_acc_list_temp))
                data_list.append(np.array(data_list_temp))
                random_acc_list_temp = []
                title_list.append(row[0])

            else:  # data
                random_acc_list_temp.append(float(row[1]))
                data_list_temp.append(int(row[-1]))

        random_acc_list.append(np.array(random_acc_list_temp))

    csv_data = {
        "data_name": data_name,
        "titles": title_list,
        "data_acc": random_acc_list,
        "data_size": data_list[0],
    }
    return csv_data


def plot_shaded_std_fig(figure, data, color_pallete):

    data_acc = np.array(data['data_acc'])
    i_plots = 1
    # get data to plot
    mean = np.mean(data_acc, axis=0)
    sem = np.std(data_acc, axis=0)/data_acc.shape[0]**0.5
    x_data = data['data_size']
    color = color_pallete[i_plots]
    # label = data['graph_name'] + ': ' + plots[i_plots].capitalize()

    # plot it
    plt.fill_between(x_data, mean-sem, mean+sem, alpha=0.5, color=color, figure=figure)
    plt.plot(x_data, mean, color=color, figure=figure, alpha=0.75)
    # plt.ylim((0.65, 1.00))

    plt.title('Index from CART to CNN, test accuracy')
    plt.grid(which='both', axis='both')
    plt.xlabel('Number of training samples')
    plt.ylabel('Test data acc [%]')
    # plt.xticks(ticks=x_data, fontsize=x_fontsize)
    plt.tight_layout()


def main():

    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    csv_test_cnn_path = 'C:\\Users\\steve\\Desktop\\projects_software\\active-learning-prototypes\\results\\' \
                        'unreduced_mnist\\from_cart\\cnn_qbc\\all.csv'

    my_dpi = 250
    plt.interactive(True)
    figure = plt.figure(figsize=(2000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    data = csv_train_reader(csv_test_cnn_path, 'unreduced_MNIST')
    plot_shaded_std_fig(figure, data, color_palette)
    plt.tight_layout()

    a = 1


if __name__ == '__main__':
    main()
