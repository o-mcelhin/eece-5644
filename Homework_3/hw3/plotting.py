import matplotlib.pyplot as plt
import numpy as np


def plot_Data(data, labels, c):
    fig, axs = plt.subplots()
    for j in range(c):
        l_idx = np.where(labels[0, :] == j)[0]
        axs.plot(data[0, l_idx], data[1, l_idx], '.', markersize=10)


def plot_results(data, labels_t, labels_o, n_class):
    # Fig init
    fig, axs = plt.subplots()
    # Class by class
    format = ['b.', 'g.', 'c.', 'k.']
    for k in range(n_class):
        idx = np.where((labels_t[0, :] == k) & (labels_o == k))[0]
        axs.plot(data[0, idx], data[1, idx], format[k])
    # Plot FA
    idx = np.where(labels_t[0, :] != labels_o)
    axs.plot(data[0, idx], data[1, idx], 'rx')

    return
