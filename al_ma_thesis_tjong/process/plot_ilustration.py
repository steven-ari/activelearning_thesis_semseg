import matplotlib.pyplot as plt
import os
from os.path import dirname as dr, abspath
import numpy as np


def plot_unlabeled_lots():
    dpi = 100
    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'for_ppt')
    plt.interactive(True)

    # prepare figure
    figure = plt.figure(figsize=(25, 15),  dpi=dpi)
    random_all = np.random.rand(2, int(200*7.5))
    plt.scatter(x=random_all[0, :], y=random_all[1, :], c='#606060', marker='x')
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    a = 1


# plot illustration of training active learning vs random
def plot_qbc_example():
    dpi = 100
    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'for_ppt')
    plt.interactive(True)

    # prepare figure
    figure = plt.subplots(nrows=2, ncols=2, figsize=(12, 6),  dpi=dpi)
    ax_all = figure[0].axes[0]
    ax_gt = figure[0].axes[1]
    ax_random = figure[0].axes[2]
    ax_active = figure[0].axes[3]

    # random points, [0, :] for x; [1, :] for y, sorted for x
    random_all = np.random.rand(2, 200)
    random_all[0, :] = np.sort(random_all[0, :])
    random_all[1, :] = random_all[1, np.argsort(random_all[0, :])]

    # plot random unlabelled
    ax_all.scatter(x=random_all[0, :], y=random_all[1, :], c='#606060', marker='x')
    ax_all.set_title('Unlabeled data', fontsize=16)
    ax_all.set_autoscale_on(False)
    # make it pretty

    # plot all labelled
    idx_cat1 = np.argwhere(random_all[0, :] < 0.5)
    idx_cat2 = np.argwhere(random_all[0, :] > 0.5)
    # give overlap
    idx_temp = idx_cat1[-6:-1].copy()
    idx_cat1[-6:-1] = idx_cat2[0:5]
    idx_cat2[0:5] = idx_temp.copy()
    # plot
    ax_gt.scatter(x=random_all[0, idx_cat1], y=random_all[1, idx_cat1], c=color_palette[0], marker="o")
    ax_gt.scatter(x=random_all[0, idx_cat2], y=random_all[1, idx_cat2], c=color_palette[1], marker="s")
    # plot boundary
    ax_gt.set_autoscale_on(False)
    # ax_gt.axvline(x=0.5, c=color_palette[2])
    ax_gt.set_title('If All Data Labeled', fontsize=16)
    # make it pretty

    # plot random training, skew towards cat 1
    skewer_cat1 = idx_cat1[np.intersect1d(np.argwhere(random_all[0, idx_cat1] > 0.35),
                                          np.argwhere(random_all[1, idx_cat1] < 0.25))].squeeze()
    skewer_cat2 = idx_cat2[np.intersect1d(np.argwhere(random_all[0, idx_cat2] < 0.65),
                                          np.argwhere(random_all[1, idx_cat2] > 0.75))].squeeze()
    idx_cat1_random = np.random.randint(low=0, high=idx_cat1.max(), size=30-min(skewer_cat1.size, 8))
    idx_cat2_random = np.random.randint(low=idx_cat2.min(), high=idx_cat2.max(), size=30-min(skewer_cat2.size, 8))
    idx_cat1_rand = np.concatenate([idx_cat1_random, skewer_cat1[0:(min(skewer_cat1.size, 8))]])  # 5 skewer data
    idx_cat2_rand = np.concatenate([idx_cat2_random, skewer_cat2[0:(min(skewer_cat2.size, 8))]])  # 5 skewer data
    # plot unlabeled
    ax_random.scatter(x=random_all[0, np.setdiff1d(range(200), [idx_cat1_rand, idx_cat2_rand])],
                      y=random_all[1, np.setdiff1d(range(200), [idx_cat1_rand, idx_cat2_rand])],
                      c='#606060', marker='x')
    # plot labeled
    ax_random.scatter(x=random_all[0, idx_cat1_rand], y=random_all[1, idx_cat1_rand],
                      c=color_palette[0], marker="o")
    ax_random.scatter(x=random_all[0, idx_cat2_rand], y=random_all[1, idx_cat2_rand],
                      c=color_palette[1], marker="s")
    # plot boundary
    ax_random.set_autoscale_on(False)
    # ax_random.plot([0.55, 0.45], [-0.1, 1.1], c=color_palette[2])
    ax_random.set_title('Trained with random', fontsize=16)
    # make it pretty

    # plot active learning
    idx_al1 = idx_cat1[-30:].squeeze()
    idx_al2 = idx_cat2[0:30].squeeze()
    # plot unlabeled
    ax_active.scatter(x=random_all[0, np.setdiff1d(range(200), [idx_al1, idx_al2])],
                      y=random_all[1, np.setdiff1d(range(200), [idx_al1, idx_al2])], c='#606060', marker='x')
    # plot labeled
    ax_active.scatter(x=random_all[0, idx_al1], y=random_all[1, idx_al1], c=color_palette[0], marker="o")
    ax_active.scatter(x=random_all[0, idx_al2], y=random_all[1, idx_al2], c=color_palette[1], marker="s")
    # plot boundary
    ax_active.set_autoscale_on(False)
    # ax_active.plot([0.49, 0.51], [-0.1, 1.1], c=color_palette[2])
    ax_active.set_title('Trained with active learning', fontsize=16)

    # make it pretty
    plt.tight_layout()
    ax_all.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_gt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_random.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_active.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # save fig
    name = 'random_vs_al_example'
    file_name = os.path.join(plot_path, (name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)


def sample_transfer():

    dpi = 150
    np.random.seed(10)
    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    plt.interactive(True)
    figure = plt.subplots(nrows=1, ncols=2, figsize=(1800/dpi, 900/dpi), dpi=dpi)
    ax_step = figure[0].axes[0]
    ax_all = figure[0].axes[1]

    # random points, [0, :] for x; [1, :] for y, sorted for x
    random_all = np.random.rand(2, 200)
    random_all[0, :] = np.sort(random_all[0, :])
    random_all[1, :] = random_all[1, np.argsort(random_all[0, :])]

    # plot chronological labelling
    ax_step.scatter(x=random_all[0, :], y=random_all[1, :], c='#606060', marker='x', alpha=0.5)
    idx_cat1 = np.argwhere((random_all[0, :] < 0.2) & (random_all[1, :] > 0.8))
    idx_cat2 = np.argwhere((random_all[0, :] > 0.8) & (random_all[1, :] < 0.8) & (random_all[1, :] > 0.6))
    idx_cat3 = np.argwhere((random_all[0, :] > 0.8) & (random_all[1, :] < 0.2))
    idx_cat4 = np.argwhere((random_all[0, :] > 0.2) & (random_all[0, :] < 0.5) &
                           (random_all[1, :] > 0.3) & (random_all[1, :] < 0.5))

    ax_step.scatter(x=random_all[0, idx_cat1], y=random_all[1, idx_cat1], c=(254/255, 90/255, 28/255), marker="s")
    ax_step.scatter(x=random_all[0, idx_cat2], y=random_all[1, idx_cat2], c=(254/255, 124/255, 34/255), marker="s")
    ax_step.scatter(x=random_all[0, idx_cat3], y=random_all[1, idx_cat3], c=(255/255, 158/255, 40/255), marker="s")
    ax_step.scatter(x=random_all[0, idx_cat4], y=random_all[1, idx_cat4], c=(255/255, 225/255, 52/255), marker="s")

    # plot all labeled
    ax_all.scatter(x=random_all[0, :], y=random_all[1, :], c='#606060', marker='x', alpha=0.5)
    ax_all.scatter(x=random_all[0, idx_cat1], y=random_all[1, idx_cat1], c=color_palette[0], marker="s")
    ax_all.scatter(x=random_all[0, idx_cat2], y=random_all[1, idx_cat2], c=color_palette[0], marker="s")
    ax_all.scatter(x=random_all[0, idx_cat3], y=random_all[1, idx_cat3], c=color_palette[0], marker="s")
    ax_all.scatter(x=random_all[0, idx_cat4], y=random_all[1, idx_cat4], c=color_palette[0], marker="s")

    # make pretty
    plt.tight_layout()
    ax_step.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_all.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


# plot illustration of confusion within a committee
def plot_qbc_confusion():
    dpi = 100
    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'for_ppt')
    plt.interactive(True)

    # prepare figure
    figure = plt.figure(figsize=(9, 3), dpi=dpi)

    # random points, [0, :] for x; [1, :] for y, sorted for x
    random_all_1 = np.random.rand(2, 60)
    random_all_1[0, :] = np.sort(random_all_1[0, :])*0.43
    random_all_2 = np.random.rand(2, 60)
    random_all_2[0, :] = (np.sort(random_all_2[0, :])*0.43)+0.57
    random_all_conf = np.random.rand(2, 20)
    random_all_conf[0, :] = (np.sort(random_all_conf[0, :])*0.1)+0.45

    # plot all labelled
    plt.scatter(x=random_all_1[0, :], y=random_all_1[1, :], c=color_palette[0], marker="o")
    plt.scatter(x=random_all_2[0, :], y=random_all_2[1, :], c=color_palette[1], marker="s")
    plt.scatter(x=random_all_conf[0, :], y=random_all_conf[1, :], c='k', marker="$?$")

    # plot boundary lines
    plt.autoscale(False)
    plt.plot([0.45, 0.53], [-0.12, 1.06], c=color_palette[2])
    plt.plot([0.42, 0.49], [-0.1, 1.2], c=color_palette[2])
    plt.plot([0.58, 0.42], [-0.1, 1.2], c=color_palette[2])
    plt.plot([0.58, 0.54], [-0.1, 1.2], c=color_palette[2])
    plt.plot([0.5, 0.5], [-0.1, 1.2], c=color_palette[2])

    # make it pretty
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    # plt.title('QBC: Versions within committee', fontsize=16)
    plt.tight_layout()

    # save fig
    name = 'qbc_confusion'
    file_name = os.path.join(plot_path, (name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)


# plot illustration for qbc with diversity
def plot_diversity():

    dpi = 100
    color_palette = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'for_ppt')
    plt.interactive(True)

    # prepare figure
    figure = plt.figure(figsize=(12, 7), dpi=dpi)

    # prepare random data
    rand1_x = np.random.normal(loc=2.0, scale=0.5, size=200)
    rand1_y = np.random.normal(loc=5.0, scale=0.35, size=200)
    rand1 = np.array([rand1_x, rand1_y])
    rand2_x = np.random.normal(loc=5.0, scale=0.25, size=200)
    rand2_y = np.random.normal(loc=5.0, scale=0.5, size=200)
    rand2 = np.array([rand2_x, rand2_y])
    rand3_x = np.random.normal(loc=9.0, scale=0.5, size=200)
    rand3_y = np.random.normal(loc=6.0, scale=0.45, size=200)
    rand3 = np.array([rand3_x, rand3_y])
    rand4_x = np.random.normal(loc=3.0, scale=0.5, size=200)
    rand4_y = np.random.normal(loc=9.0, scale=0.5, size=200)
    rand4 = np.array([rand4_x, rand4_y])
    rand5_x = np.random.normal(loc=6.5, scale=0.75, size=200)
    rand5_y = np.random.normal(loc=7.5, scale=0.5, size=200)
    rand5 = np.array([rand5_x, rand5_y])

    plt.scatter(x=rand1[0, :], y=rand1[1, :], c=color_palette[0], marker='o')
    plt.scatter(x=rand2[0, :], y=rand2[1, :], c=color_palette[1], marker='o')
    plt.scatter(x=rand3[0, :], y=rand3[1, :], c=color_palette[2], marker='o')
    plt.scatter(x=rand4[0, :], y=rand4[1, :], c=color_palette[3], marker='o')
    plt.scatter(x=rand5[0, :], y=rand5[1, :], c=color_palette[4], marker='o')

    # make it pretty
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    plt.tick_params(axis='both', which='both', direction='in')
    plt.tight_layout()

    # save fig
    name = 'qbc_w_diversity'
    file_name = os.path.join(plot_path, (name + '.png'))
    plt.savefig(file_name, format='png', dpi=300)


if __name__ == '__main__':
    plot_unlabeled_lots()