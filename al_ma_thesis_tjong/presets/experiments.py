import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import scipy.stats as stats
import math


def main():
    n = 256
    dr = 1-0.9

    k = np.arange(n)
    pmf = [comb(n, x)*(dr**x)*((1-dr)**(n-x)) for x in k]

    # plot just pmf
    plt.interactive(True)
    my_dpi = 175
    color_palette = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
    figure, ax_iou = plt.subplots(figsize=(1900 / my_dpi, 700 / my_dpi), dpi=my_dpi)
    ax_iou.scatter(k, pmf, color=color_palette[0])

    # draw maximum line
    y_lim = ax_iou.get_ylim()
    ax_iou.vlines(n*dr, -0.5, np.max(pmf)*1.2, color='tab:red', linestyles='--')
    ax_iou.text(26, np.max(pmf)*0.76, '$E(X)$=25.6', color='tab:red', fontsize=12)

    # plot pseudo normal
    mean = n*dr
    sigma = math.sqrt(n*(dr)*(1-dr))
    x = np.linspace(mean - 1 * sigma, mean + 1 * sigma, 100)
    ax_iou.fill_between(x, np.zeros_like(x), stats.norm.pdf(x, mean, sigma), alpha=0.3, color='tab:blue')
    x = np.linspace(mean - 2 * sigma, mean + 2 * sigma, 100)
    ax_iou.fill_between(x, np.zeros_like(x), stats.norm.pdf(x, mean, sigma), alpha=0.3, color='tab:blue')
    ax_iou.text(26.3, np.max(pmf)*0.01, '$\sigma$=4.8', color='white', fontsize=12)

    ax_iou.set_xlim((0, 60))
    ax_iou.set_ylim(y_lim)
    ax_iou.set_xlabel('Number of Active Nodes')
    ax_iou.set_ylabel('PMF Value')
    plt.tight_layout()

    a = 1






if __name__ == '__main__':
    main()