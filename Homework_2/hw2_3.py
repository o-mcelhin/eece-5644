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
        # Get bootstrap samples for data
        data_t, data_v = bootstrap(data, samples, (3*samples)//4)
        for model in range(1, 7):
            # Initialize variables
            mu_init = np.zeros((1, 2, model))
            cov = np.zeros((2, 2, model))
            # Initial guess
            hand = list(range(1, samples))
            random.shuffle(hand)
            # Pick random point as initial mean
            for current_mu in range(model):
                mu_init[:, :, current_mu] = data[:, hand[current_mu]]
            # Find L2 norm from each point to each mean to attribute to groups
            norms = np.zeros((data_t.shape[1], model))
            group = np.zeros((1, data_t.shape[1]))
            for nums, cur_data in enumerate(data_t.T):
                for model_sub in range(model):
                    norms[nums, model_sub] = np.linalg.norm(np.subtract(cur_data, mu_init[:, :, model_sub]))
                    hold = np.where(norms[nums, :] == np.amin(norms[nums, :]))[0]
                group[0, nums] = np.where(norms[nums, :] == np.amin(norms[nums, :]))[0]
            # Covariance of each group
            alpha = np.divide(np.ones((1, model)), model)
            for model_sub in range(model):
                idcs = np.where(group[0, :] == model_sub)[0]
                hold = data_t[:, idcs]
                cov[:, :, model_sub] = np.cov(hold, rowvar=True)
            converge = 20
            mu = mu_init
            while converge >= 0.05:
                likelihood = np.zeros((model, data_t.shape[1]))
                for model_sub in range(model):
                    l = []
                    for cur_data in data_t.T:
                        l.append(alpha[0, model_sub] * evaluate_gaussian(cur_data, mu[:, :, model_sub], cov[:, :, model_sub]))
                    l = np.asarray(l)
                    likelihood[model_sub, :] = l.T
                hold = np.sum(likelihood, 0)
                pr_l = np.zeros((model, data_t.shape[1]))
                for z in range(likelihood.shape[0]):
                    pr_l[z, :] = np.divide(likelihood[z, :], np.sum(likelihood, 0))
                # Update prior
                alpha_new = np.mean(pr_l, 1)
                # Weight matrix
                w = np.zeros((model, data_t.shape[1]))
                for z in range(likelihood.shape[0]):
                    w[z, :] = np.divide(pr_l[z, :], np.sum(pr_l, 1))
                # Update mu
                mu_new = np.dot(data_t, w.T)
                # update cov
                cov_new = np.zeros((2, 2, model))
                for z in range(likelihood.shape[0]):
                    s1 = np.square(np.subtract(data_t.T, mu_new[:, z]))
                    hold = mlib.repmat(w[k, :], 2, 1)
                    s2 = np.multiply(mlib.repmat(w[k, :], 2, 1), s1.T)
                    cov_new[:, :, z] = np.dot(s2, s1) + 1e-7*np.identity(2)
                # Find changes
                d_mu = np.amax(np.subtract(mu, mu_new))
                d_cov = np.amax(np.subtract(cov, cov_new))
                d_alpha = np.amax((np.subtract(alpha, alpha_new)))
                converge = np.amax([d_alpha, d_mu, d_cov])
                # Update model
                for z in range(model):
                    mu[:, :, z] = mu_new[:, z]
                    cov[:, :, z] = cov_new[:, :, z]
                alpha = np.array([alpha_new])








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
    for k in samples:
        for m in range(100):
            theta = em_gmm(alpha, mu, cov, k)
