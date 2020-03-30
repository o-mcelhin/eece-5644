import numpy as np
import math
import matplotlib.pyplot as plt
from hw3 import plotting
import numpy.matlib as mlib


def generate_MultiRing(nClass, nSamples):
    """
        This function generates n 'ring' distributions. The radius of each class is determined from a gamma distribution
    and the angle is determined from a uniform distribution
    """
    # imports
    c = nClass
    n = nSamples
    # Determine class labels
    tau = np.linspace(0, 1, c+1)
    labels = np.zeros((1, nSamples))
    # Uniform distribution to randomly assign class labels
    dist = np.random.uniform(0, 1, (1, nSamples))
    # Distribute class labels
    for j in range(c):
        l_idx = np.where((tau[j] <= dist[0, :]) & (dist[0, :] < tau[j+1]))[0]
        temp = np.multiply(j, np.ones((1, len(l_idx))))
        labels[0, l_idx] = temp
    # Generate parameters of Gamma distribution
    a = np.power(np.linspace(1, c, c), 3)
    b = np.multiply(2, np.ones((1, c)))
    # Generate angle from uniform distribution
    angle = np.multiply(2*math.pi, np.random.uniform(0, 1, (1, n)))
    radius = np.zeros((1, n))
    for j in range(c):
        l_idx = np.where(labels[0, :] == j)[0]
        radius[0, l_idx] = np.random.gamma(a[j], b[0, j], len(l_idx))
    # Generate data samples
    data = np.zeros((2, n))
    data[0, :] = np.multiply(radius, np.cos(angle))
    data[1, :] = np.multiply(radius, np.sin(angle))

    # Plotting
    plotting.plot_Data(data, labels, c)

    return data, labels


def reshape_kfold(k, data_train, labels_train):
    # Find how many samples are given
    N = labels_train.shape[1]
    # How many samples per fold
    N_fold = N//k
    # How many folds
    K_fold = N//N_fold
    # Init array
    data_folded = np.zeros((data_train.shape[0], N_fold, K_fold))
    labels_folded = np.zeros((1, N_fold, K_fold))
    for j in range(K_fold):
        data_folded[:, :, j] = data_train[:, (N_fold*j):(N_fold*(j+1))]
        labels_folded[:, :, j] = labels_train[0, (N_fold*j):(N_fold*(j+1))]

    return data_folded, labels_folded
