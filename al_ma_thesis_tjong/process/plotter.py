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


def csv_train_reader(csv_path, data_name):
    title_list = []
    acc_list = []
    acc_avg_list = []
    acc_std_list = []
    batch_size_list = []
    committee_vote_list = []

    batch_size = []
    acc_test = []
    acc_avg = []
    acc_std = []
    committee_vote = []

    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:  # first title
                title_list.append(row[0])
                line_count += 1
            elif row.__len__() == 1:  # title again
                title_list.append(row[0])
                acc_list.append(np.stack(acc_test, axis=0))  # append to bigger list
                acc_avg_list.append(np.asarray(acc_avg))
                acc_std_list.append(np.asarray(acc_std))
                batch_size_list.append(np.asarray(batch_size))
                committee_vote_list.append(np.asarray(committee_vote))

                acc_test = []  # clean temporary list
                acc_avg = []
                acc_std = []
                batch_size = []
                committee_vote = []
                line_count += 1
            else:  # acc numbers
                acc_test.append(np.asarray([100 * float(acc) for acc in row[1:-3]]))  # append to temp list
                acc_avg.append(float(row[-2][:-1]))
                acc_std.append(float(row[-1][:-1]))
                batch_size.append(int(row[0]))
                committee_vote.append(float(row[-3][:-1]))
                line_count += 1
        acc_list.append(np.stack(acc_test, axis=0))
        acc_avg_list.append(np.asarray(acc_avg))
        acc_std_list.append(np.asarray(acc_std))
        batch_size_list.append(np.asarray(batch_size))
        committee_vote_list.append(np.asarray(committee_vote))

    csv_data = {
        "data_name": data_name,
        "titles": title_list,
        "acc_list": acc_list,
        "acc_avg_list": acc_avg_list,
        "acc_std_list": acc_std_list,
        "batch_size_list": batch_size_list,
        "committee_vote_list": committee_vote_list,
    }
    return csv_data


def shaded_std_line(idx_plot, csv_dict, figure, label, color):
    n_plots = 10
    std = csv_dict["committee_vote_std_list"][idx_plot]
    # indices = [0] + list(range(int(std.shape[0] / n_plots) - 1, std.shape[0], int(std.shape[0] / n_plots)))
    indices = list(range(std.shape[0]))
    batch_size = csv_dict["batch_size_list"][idx_plot][indices]
    committee_vote = csv_dict["committee_vote_list"][idx_plot][indices]
    std = csv_dict["committee_vote_std_list"][idx_plot][indices]/len(csv_dict["committee_vote_std_list"][idx_plot][indices])**0.5
    plt.fill_between(batch_size, committee_vote - std, committee_vote + std, alpha=0.7, color=color, figure=figure)
    plt.plot(batch_size, committee_vote, color=color, label=label, figure=figure)


def plot_shaded_std(train_idx, csv_dict, color_palette, my_dpi):
    plt.interactive(True)
    name = 'entire_data'
    max_val = True
    legend_loc = 7
    x_fontsize = 10
    plot_max = False

    figure = plt.figure(figsize=(1200 / my_dpi, 1200 / my_dpi), dpi=my_dpi)
    shaded_std_line(train_idx[2], csv_dict, figure, 'QBC entropy + diversity', color_palette[0])
    shaded_std_line(train_idx[0], csv_dict, figure, 'QBC: Vote', color_palette[1])
    shaded_std_line(train_idx[1], csv_dict, figure, 'without AL', color_palette[2])

    '''# draw line for max acc
    if max_val:
        color_list = [1, 2, 0]
        for i_line in range(3):
            y_lim = figure.axes[0].get_ylim()
            # get x position of max value
            max_acc = csv_dict["committee_vote_list"][train_idx[i_line]].max()
            x_pos = csv_dict["batch_size_list"][train_idx[i_line]][csv_dict["committee_vote_list"][train_idx[i_line]].argmax()]
            plt.axvline(x=x_pos, c=color_palette[color_list[i_line]], ls='dashed')
            # prevent text goes outside axes
            if x_pos < (figure.axes[0].get_xlim()[1]*0.6):
                h_align = 'right'
                x_text = x_pos * 0.98
            else:
                h_align = 'right'
                x_text = x_pos*0.98
            # text
            t = plt.text(x=x_text, y=y_lim[0] + (y_lim[1] - y_lim[0])*(0.25 - i_line*0.1),
                         s=("{:.2f}".format(max_acc) + ' %; data = ' + str(x_pos)), c=color_palette[color_list[i_line]],
                         ha=h_align)
            t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))'''

    if plot_max:
        acc_all_data = csv_dict['committee_vote_list'][8][-1]
        plt.axhline(y=acc_all_data, color='r', linestyle='dashed', alpha=0.6)
        x_lim = figure.axes[0].get_xlim()
        text = ('Acc, entire dataset: ')
        plt.text(x=x_lim[0] + (x_lim[1] - x_lim[0])*0.05, y=acc_all_data,
                 s=(text + "{:.2f}".format(acc_all_data) + ' %'), c='r',
                 ha='left')

    n_plots = 10
    # for x_ticks
    indices = [0] + list(range(int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots) - 1,
                               csv_dict["batch_size_list"][train_idx[1]].shape[0],
                               int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots)))
    plt.title(name.replace("_", " ").capitalize() + ' size, ' + csv_dict["data_name"])
    plt.grid(which='both', axis='both')
    plt.xlabel('Number of training samples')
    plt.legend(loc=legend_loc)
    plt.ylabel('Test data acc [%]')


    # safe fig
    if csv_dict["batch_size_list"][train_idx[1]][indices][0] == 1000:
        file_name = name + '_' + str(1000)
        indices = [0] + list(range(int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots) - 1,
                                   csv_dict["batch_size_list"][train_idx[1]].shape[0],
                                   int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots)))
    elif csv_dict["batch_size_list"][train_idx[1]][indices][0] == 3000:
        file_name = name + '_' + str(3000)
        indices = [0] + list(range(int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots) - 1,
                                   csv_dict["batch_size_list"][train_idx[1]].shape[0],
                                   int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots)))

    plt.xticks(ticks=csv_dict["batch_size_list"][train_idx[1]][indices], fontsize=x_fontsize, rotation=45)
    plt.tight_layout()
    result_dir = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', csv_dict["data_name"].replace(" ", "_"))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    file_name = os.path.join(result_dir, (file_name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)


def plot_shaded_std_single(train_idx, csv_dict, color_palette, my_dpi):
    plt.interactive(True)
    name = 'entire_data'
    max_val = True
    legend_loc = 7
    x_fontsize = 10
    plot_max = False

    figure = plt.figure(figsize=(850 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    shaded_std_line(train_idx[0], csv_dict, figure, 'QBC entropy', color_palette[1])

    # draw line for max acc
    if max_val:
        i_line = 0
        i_color = 1
        y_lim = figure.axes[0].get_ylim()
        # get x position of max value
        max_acc = csv_dict["committee_vote_list"][train_idx[i_line]].max()
        x_pos = csv_dict["batch_size_list"][train_idx[i_line]][csv_dict["committee_vote_list"][train_idx[i_line]].argmax()]
        plt.axvline(x=x_pos, c=color_palette[i_color], ls='dashed')
        # prevent text goes outside axes
        if x_pos < (figure.axes[0].get_xlim()[1]*0.6):
            h_align = 'left'
            x_text = x_pos*1.03
        else:
            h_align = 'right'
            x_text = x_pos*0.98
        # text
        t = plt.text(x=x_text, y=y_lim[0] + (y_lim[1] - y_lim[0])*(0.25 - i_line*0.1),
                     s=("{:.2f}".format(max_acc) + ' %; data = ' + str(x_pos)), c=color_palette[i_color],
                     ha=h_align)
        t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))

    if plot_max:
        acc_all_data = csv_dict['committee_vote_list'][8][-1]
        plt.axhline(y=acc_all_data, color='r', linestyle='dashed', alpha=0.6)
        x_lim = figure.axes[0].get_xlim()
        text = ('Acc, entire dataset: ')
        plt.text(x=x_lim[0] + (x_lim[1] - x_lim[0])*0.05, y=acc_all_data,
                 s=(text + "{:.2f}".format(acc_all_data) + ' %'), c='r',
                 ha='left')

    n_plots = 10
    # for x_ticks
    indices = [0] + list(range(int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots) - 1,
                               csv_dict["batch_size_list"][train_idx[1]].shape[0],
                               int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots)))
    plt.title(name.replace("_", " ").capitalize() + ' size, ' + csv_dict["data_name"])
    plt.grid(which='both', axis='both')
    plt.xlabel('Number of training samples')
    plt.legend(loc=legend_loc)
    plt.ylabel('Test data acc [%]')


    # safe fig
    if csv_dict["batch_size_list"][train_idx[1]][indices][0] == 1000:
        file_name = name + '_' + str(1000)
        indices = [0] + list(range(int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots) - 1,
                                   csv_dict["batch_size_list"][train_idx[1]].shape[0],
                                   int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots)))
    elif csv_dict["batch_size_list"][train_idx[1]][indices][0] == 3000:
        file_name = name + '_' + str(3000)
        indices = [0] + list(range(int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots) - 1,
                                   csv_dict["batch_size_list"][train_idx[1]].shape[0],
                                   int(csv_dict["batch_size_list"][train_idx[1]].shape[0] / n_plots)))

    plt.xticks(ticks=csv_dict["batch_size_list"][train_idx[1]][indices], fontsize=x_fontsize)
    plt.tight_layout()
    result_dir = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', csv_dict["data_name"].replace(" ", "_"))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    file_name = os.path.join(result_dir, (file_name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)

'''def plot_shaded_std(train_idx, csv_dict, color_palette, my_dpi):

    if train_idx[0] == 0:
        name = 'minimal_data'
        max_val = False
        legend_loc = 4
        x_fontsize = 12
        plot_max = False
    elif train_idx[0] == 3:
        name = 'medium_data'
        max_val = False
        legend_loc = 4
        x_fontsize = 12
        plot_max = False
    elif train_idx[0] == 6:
        name = 'entire_data'
        max_val = True
        legend_loc = 7
        x_fontsize = 10
        plot_max = False
    elif train_idx[0] == 9:
        name = 'half_dataset'
        max_val = True
        legend_loc = 7
        x_fontsize = 10
        plot_max = True

    figure = plt.figure(figsize=(700 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    shaded_std_line(train_idx[0], csv_dict, figure, 'QBC entropy + diversity', color_palette[0])
    shaded_std_line(train_idx[1], csv_dict, figure, 'QBC entropy', color_palette[1])
    shaded_std_line(train_idx[2], csv_dict, figure, 'Random', color_palette[2])

    # draw line for max acc
    if max_val:
        for i_line in range(3):
            y_lim = figure.axes[0].get_ylim()
            # get x position of max value
            max_acc = csv_dict["committee_vote_list"][train_idx[i_line]].max()
            x_pos = csv_dict["batch_size_list"][train_idx[i_line]][csv_dict["committee_vote_list"][train_idx[i_line]].argmax()]
            plt.axvline(x=x_pos, c=color_palette[i_line], ls='dashed')
            # prevent text goes outside axes
            if x_pos < (figure.axes[0].get_xlim()[1]*0.6):
                h_align = 'left'
                x_text = x_pos*1.03
            else:
                h_align = 'right'
                x_text = x_pos*0.98
            # text
            t = plt.text(x=x_text, y=y_lim[0] + (y_lim[1] - y_lim[0])*(0.25 - i_line*0.1),
                         s=("{:.2f}".format(max_acc) + ' %; data = ' + str(x_pos)), c=color_palette[i_line],
                         ha=h_align)
            t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))

    if plot_max:
        acc_all_data = csv_dict['committee_vote_list'][8][-1]
        plt.axhline(y=acc_all_data, color='r', linestyle='dashed', alpha=0.6)
        x_lim = figure.axes[0].get_xlim()
        text = ('Acc, entire dataset: ')
        plt.text(x=x_lim[0] + (x_lim[1] - x_lim[0])*0.05, y=acc_all_data,
                 s=(text + "{:.2f}".format(acc_all_data) + ' %'), c='r',
                 ha='left')


    plt.title(name.replace("_", " ").capitalize() + ' size, ' + csv_dict["data_name"])
    plt.grid(which='both', axis='both')
    plt.xticks(ticks=csv_dict["batch_size_list"][train_idx[1]], fontsize=x_fontsize)
    plt.xlabel('Number of training data []')
    plt.legend(loc=legend_loc)
    plt.ylabel('Test data acc [%]')
    plt.tight_layout()

    # safe fig
    result_dir = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', csv_dict["data_name"].replace(" ", "_"))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    file_name = os.path.join(result_dir, (name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)'''


def plot_hist(idx_plot, idx_batch, csv_dict, figure, color):
    plt.bar(x=range(1, csv_dict["acc_list"][idx_plot][idx_batch].__len__() + 1),
            height=csv_dict["acc_list"][idx_plot][idx_batch], figure=figure, color=color)
    plt.xticks(ticks=range(1, csv_dict["acc_list"][idx_plot][idx_batch].__len__() + 1))


# figure = plt.figure(figsize=(700/my_dpi, 700/my_dpi), dpi=my_dpi)
# plot_hist(idx_plot=1, idx_batch=1, csv_dict=mnist_reduced, figure=figure, color=color_palette[0])


# plot vote entropy
def entropy_plotter(csv_path, color_palette, my_dpi):

    plt.interactive(True)
    title_list = ['vote entropy, acc=30%', 'vote entropy, acc=50%', 'vote entropy, acc=80%']
    plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'for_ppt')

    entropy_list = []
    acc_list = []
    acc_flag = True

    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if acc_flag:
                acc = np.array(row).astype(float)
                acc_list.append(acc)
                acc_flag = False
            else:
                en = np.array(row).astype(float)
                entropy_list.append(en)
                acc_flag = True

    # plot it
    fig = plt.figure(figsize=(1000 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 6], height_ratios=[6, 1])
    ax_en = plt.subplot(gs[0])
    ax_data = plt.subplot(gs[1])
    ax_acc = plt.subplot(gs[3])
    axes = [ax_en, ax_data, ax_acc]

    for i_en in range(3):
        ax_data.scatter(x=acc_list[i_en][0:500]*100, y=entropy_list[i_en][0:500], c=color_palette[i_en],
                    alpha=0.4, label=title_list[i_en])
        # draw data distribution, acc
        sns.kdeplot(acc_list[i_en][0:500]*100, color=color_palette[i_en], ax=ax_acc, shade=True)
        # draw data distribution, entropy
        sns.kdeplot(entropy_list[i_en][0:500], color=color_palette[i_en], ax=ax_en, vertical=True, shade=True)

    # plot maximal entropy value possible
    max_en = -1*(10*0.1*math.log(0.1)/math.log(2))
    ax_data.axhline(y=max_en, color='r', linestyle='dashed')

    # plot ideal line entropy value possible
    ax_data.plot([0, 10, 100], [(math.log(9)/math.log(2)), max_en, 0], color='#003399', linestyle='dashed')

    # make data looks pretty
    ax_data.legend(loc=(0.61, 0.71))
    ax_data.grid(which='both')
    ax_data.minorticks_on
    ax_data.set_title('Vote Entropy on 500 random MNIST data, Committee: 20 models', fontsize=15)
    ax_data.set_xlabel('Model with correct prediction in committee [%]', fontsize=14)
    ax_data.set_ylabel('Vote Entropy []', fontsize=14)

    # make entropy dist look pretty
    ax_en.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_en.set_ylim(ax_data.get_ylim())
    ax_en.grid(which='both', axis='y')

    # make acc dist look pretty
    ax_acc.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_acc.set_xlim(ax_data.get_xlim())
    ax_acc.grid(which='both', axis='x')

    plt.tight_layout()
    # save fig
    name = 'vote_entropy_fig'
    file_name = os.path.join(plot_path, (name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)


def cluster_size(csv_path, color_palette, my_dpi):
    title = []
    count = []
    size = []
    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:  # first title
                title = row[0]
                line_count += 1
            elif line_count == 1:  # title again
                count.append(row[0])
                size.append(row[1])

    # plot it
    count = np.array(count).astype(int)
    size = np.array(size).astype(int)
    mean = statistics.mean(size)
    var = statistics.variance(size) ** 0.5
    plt.figure(figsize=(1200 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    plt.axhline(y=mean, color='r', linestyle='dashed', alpha=0.5)
    plt.bar(x=count, height=size)

    # make it looks preety
    plt.legend()
    plt.grid(which='both')
    plt.minorticks_on

    title_text = ('On MNIST (60000 data, 20 clusters), Ideal size = ' + "{:.0f}".format(np.sum(size)/size.__len__()) +
                  '; Var size = ' + "{:.2f}".format(var))
    plt.title(title_text)
    plt.xticks(ticks=count)
    plt.xlabel('n-th cluster')
    plt.ylabel('Data in cluster []')
    plt.tight_layout()

    plt.savefig('/Users/steven_ari/Desktop/active-learning-prototypes/results/plots/cluster_size.png',
                    format='png', dpi=300)


def normal_dist(csv_path, color_palette, my_dpi):
    title = []
    acc_list = []
    batch_list = []
    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:  # first title
                title = row[0]
                line_count += 1
            else:
                batch_list.append(row[0])
                acc_list.append(np.asarray([float(val.replace(',', '.')) for val in row[1:]]))

    # plot it
    plt.figure(figsize=(1200 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    for i_batch in range(acc_list.__len__()):
        plt.hist(x=acc_list[i_batch], color=color_palette[i_batch], alpha=0.5, bins=10,
                 label=('Batch size:' + str(batch_list[i_batch])))

    # make it looks pretty
    plt.legend()
    plt.grid(which='both')
    plt.minorticks_on
    plt.title('Histogram on different batch sizes, sample size each batch=' + str(acc_list[0].__len__()))
    plt.xlabel('Acc [%]')
    plt.ylabel('Data in cluster []')
    plt.tight_layout()

    plt.savefig('/Users/steven_ari/Desktop/active-learning-prototypes/results/plots/normal_dist.png',
                format='png', dpi=300)


def avg_runs_1000(csv_dict):
    n_run_types = 3
    n_runs = int(csv_dict["committee_vote_list"].__len__()/n_run_types)

    avg_all_runs = []
    std_all_runs = []

    for i_run_types in range(0, n_run_types):
        run_pack = csv_dict["committee_vote_list"][i_run_types:
                                                   int(csv_dict["committee_vote_list"].__len__()/2):n_run_types]
        run_pack_np = np.stack(run_pack, axis=0)
        mean = np.mean(run_pack_np, axis=0)
        var = np.var(run_pack_np, axis=0)**0.5
        avg_all_runs.append(mean)
        std_all_runs.append(var)

    dict_with_runs = {
        "committee_vote_list": avg_all_runs,
        "committee_vote_std_list": std_all_runs,
        "batch_size_list": csv_dict["batch_size_list"][0:n_run_types],
        "data_name": csv_dict["data_name"]
    }

    return dict_with_runs


def avg_runs_3000(csv_dict):
    n_run_types = 3
    n_runs = int(csv_dict["committee_vote_list"].__len__()/n_run_types)

    avg_all_runs = []
    std_all_runs = []

    for i_run_types in range(0, n_run_types):
        indices = slice(i_run_types + int(csv_dict["committee_vote_list"].__len__()/2),
                        csv_dict["committee_vote_list"].__len__(),
                        n_run_types)
        run_pack = csv_dict["committee_vote_list"][indices]
        run_pack_np = np.stack(run_pack, axis=0)
        mean = np.mean(run_pack_np, axis=0)
        var = np.var(run_pack_np, axis=0)**0.5
        avg_all_runs.append(mean)
        std_all_runs.append(var)

    dict_with_runs = {
        "committee_vote_list": avg_all_runs,
        "committee_vote_std_list": std_all_runs,
        "batch_size_list": csv_dict["batch_size_list"][-4:-1],
        "data_name": csv_dict["data_name"]
    }

    return dict_with_runs


def avg_runs_3000_single(csv_dict):
    n_run_types = 1
    n_runs = int(csv_dict["committee_vote_list"].__len__()/n_run_types)

    avg_all_runs = []
    std_all_runs = []

    for i_run_types in range(0, n_run_types):
        indices = slice(i_run_types + int(csv_dict["committee_vote_list"].__len__()/2),
                        csv_dict["committee_vote_list"].__len__(),
                        n_run_types)
        run_pack = csv_dict["committee_vote_list"][indices]
        run_pack_np = np.stack(run_pack, axis=0)
        mean = np.mean(run_pack_np, axis=0)
        var = np.var(run_pack_np, axis=0)**0.5
        avg_all_runs.append(mean)
        std_all_runs.append(var)

    dict_with_runs = {
        "committee_vote_list": avg_all_runs,
        "committee_vote_std_list": std_all_runs,
        "batch_size_list": csv_dict["batch_size_list"][-4:-1],
        "data_name": csv_dict["data_name"]
    }

    return dict_with_runs



def main():
    """SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title"""

    plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots')
    csv_mnist_reduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'reduced_mnist', 'xgb_qbc.csv')
    csv_mnist_unreduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'unreduced_mnist', 'xgb_qbc.csv')
    csv_f_mnist_reduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'reduced_f_mnist', 'xgb_qbc.csv')
    csv_f_mnist_unreduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'unreduced_f_mnist', 'xgb_qbc.csv')
    csv_entropy = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've.csv')
    csv_cluster_size = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'cluster_size.csv')
    csv_normal = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'normal_dist_committee.csv')

    mnist_reduced = csv_train_reader(csv_mnist_reduced, 'encoded MNIST')
    mnist_unreduced = csv_train_reader(csv_mnist_unreduced, 'MNIST')
    f_mnist_reduced = csv_train_reader(csv_f_mnist_reduced, 'reduced_F-MNIST')
    # f_mnist_unreduced = csv_train_reader(csv_f_mnist_unreduced, 'unreduced_F-MNIST')


    '''# For testing on training data
    csv_mnist_reduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'reduced_mnist', 'xgb_qbc_train.csv')
    # csv_mnist_unreduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'unreduced_mnist', 'xgb_qbc_train.csv')
    csv_f_mnist_reduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'reduced_f_mnist', 'xgb_qbc_train.csv')
    # csv_f_mnist_unreduced = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'unreduced_f_mnist', 'xgb_qbc_train.csv')

    mnist_reduced = csv_train_reader(csv_mnist_reduced, 'reduced_MNIST_train')
    # mnist_unreduced = csv_train_reader(csv_mnist_unreduced, 'unreduced_MNIST_train')
    # f_mnist_reduced = csv_train_reader(csv_f_mnist_reduced, 'reduced_F-MNIST_train')
    # f_mnist_unreduced = csv_train_reader(csv_f_mnist_unreduced, 'unreduced_F-MNIST_train')'''

    avg_mnist_reduced_1000 = avg_runs_1000(mnist_reduced)
    # avg_f_mnist_reduced_1000 = avg_runs_1000(f_mnist_reduced)
    avg_mnist_reduced_3000 = avg_runs_3000(mnist_reduced)
    avg_f_mnist_reduced_3000 = avg_runs_3000(f_mnist_reduced)
    avg_mnist_unreduced_3000 = avg_runs_3000(mnist_unreduced)


    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    my_dpi = 180

    # plot vote entropy
    # entropy_plotter(csv_entropy, color_palette, my_dpi)
    # cluster_size(csv_cluster_size, color_palette, my_dpi)
    # normal_dist(csv_normal, color_palette, my_dpi)

    # plot_shaded_std([0, 1, 2], avg_mnist_reduced_1000, color_palette, my_dpi)
    # plot_shaded_std([0, 1, 2], avg_f_mnist_reduced_1000, color_palette, my_dpi)

    # plot_shaded_std([0, 1, 2], avg_mnist_reduced_3000, color_palette, my_dpi)
    # plot_shaded_std([0, 1, 2], avg_f_mnist_reduced_3000, color_palette, my_dpi)
    plot_shaded_std([0, 1, 2], avg_mnist_unreduced_3000, color_palette, 200)

    a = 1


if __name__ == '__main__':
    main()



