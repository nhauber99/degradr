import numpy as np
from matplotlib import pyplot as plt


def histogram_plot(histogram: np.array, log_y: bool):
    non_zero_indices = np.where(histogram != 0)[0]
    plt.xlim(min(non_zero_indices), max(non_zero_indices))
    if log_y:
        plt.semilogy(histogram)
    else:
        plt.plot(histogram)
