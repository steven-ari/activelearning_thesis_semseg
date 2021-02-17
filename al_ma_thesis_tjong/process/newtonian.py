import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.interactive(True)

    # plot loss func
    x = np.arange(-2.5, 3., 0.1)
    y = x**4 - x**2 - 6*x - 6
    plt.plot(x, y)

    # plot optim 1
    x_optim1 = np.array([-2, -1.5, -0.5, 1.8])
    y_optim1 = x_optim1 ** 4 - x_optim1 ** 2 - 6 * x_optim1 - 6
    plt.plot(x_optim1, y_optim1, 'r--', x_optim1, y_optim1, 'ro', )

    # plot optim 2
    x_optim2 = np.array([2.6, 2.2, 1.7, 0.9])
    y_optim2 = x_optim2 ** 4 - x_optim2 ** 2 - 6 * x_optim2 - 6
    plt.plot(x_optim2, y_optim2, 'g--', x_optim2, y_optim2, 'go')

    # clean plot
    plt.tight_layout()
    plt.axis('off')

    a=1

if __name__ == '__main__':
    main()