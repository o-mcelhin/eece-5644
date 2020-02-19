import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.stats
import numpy.matlib as mlib


def data_gen(alpha, mu, cov, samples):
    data = np.zeros((2, samples))
    uni = np.random.uniform(0, 1, (1, samples))
    for a in range(len(alpha)):
        prev = sum(alpha[:a])
        idcs = np.where((uni[0, :] >= prev) & (uni[0, :] <= prev+alpha[a]))[0]
        for d in idcs:
            ph = mu[0, :, a]
            data[:, d] = np.random.multivariate_normal(mu[0, :, a], cov[:, :, a])
    # plt.plot(data[0, :], data[1, :], 'bo')
    # plt.show()

    return data


def bootstrap(data, samples, n_train):
    idcs_t = np.random.randint(0, samples-1, (1, n_train))
    data_t = data[:, idcs_t[0, :]]
    idcs_v = np.random.randint(0, samples-1, (1, n_train))
    data_v = data[:, idcs_v[0, :]]

    return data_t, data_v


def em_gmm(alpha_true, mu_true, cov_true, samples):
    data = data_gen(alpha_true, mu_true, cov_true, samples)
    M = range(1, 7)
    for k in range(10):
        # Bootstrap some data
        data_t, data_v = bootstrap(data, samples, (3*samples)//5)
        for m in M:
            # Initialization
            mu = np.zeros((1, 2, m))
            cov = np.zeros((2, 2, m))
            n = data_t.shape[1]
            # Select m points to initialize parameters
            # randomize indices
            hand = list(range(0, n))
            random.shuffle(hand)
            for select in range(m):
                mu[:, :, select] = data_t[:, hand[select]].T
            # Assign each training point to a group based on distance to means
            dist = np.zeros((m, n))
            for j in range(m):
                dist[j, :] = np.linalg.norm(np.subtract(data_t, mu[:, :, j].T), axis=0)
            # Find row where each column is at min and assign as group
            mins = np.amin(dist, axis=0)
            groups = np.where(dist == mins)[0]
            ph = 1




def evaluate_gaussian(x, m, cov):
    l = (1 / (np.sqrt((2 * math.pi) ** 2 * np.linalg.det(cov)))) * math.exp(
        -0.5 * (np.dot(np.subtract(x, m).dot(np.linalg.inv(cov)), np.subtract(x, m).T)))

    return l


if __name__ == '__main__':
    # Define true parameters
    alpha = [0.2, 0.3, 0.1, 0.4]
    # Means
    mu = np.zeros((1, 2, 4))
    mu[:, :, 0] = [5, 3]
    mu[:, :, 1] = [3, -2]
    mu[:, :, 2] = [-2, 3]
    mu[:, :, 3] = [-4, -3]
    # Covariances
    cov = np.zeros((2, 2, 4))
    cov[:, :, 0] = [[1, 0.5], [0.5, 1]]
    cov[:, :, 1] = [[1, -0.25], [-0.25, 1]]
    cov[:, :, 2] = [[1, 0.1], [0.1, 1]]
    cov[:, :, 3] = [[1, 0.75], [0.75, 1]]
    # Sample steps
    samples = [10, 100, 1000]
    # Call EM for GMM 100 times
    for N in samples:
        for runs in range(100):
            theta = em_gmm(alpha, mu, cov, N)