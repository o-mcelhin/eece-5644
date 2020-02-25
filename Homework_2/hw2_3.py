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
    plt.plot(data[0, :], data[1, :], 'bo')
    plt.show()

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
    ll_mat = np.zeros((7, 10))
    for k in range(10):
        # Bootstrap some data
        data_t, data_v = bootstrap(data, samples, samples)
        for m in M:
            # Initialization
            mu = np.zeros((1, 2, m))
            cov = np.zeros((2, 2, m))
            n = data_t.shape[1]
            # Select m points to initialize parameters
            alpha = (1/m) * np.ones((1, m))
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
            groups = np.zeros((1, n))
            for j in range(n):
                groups[0, j] = np.where(mins[j] == dist[:, j])[0][0]
            hold_g = np.where(mins == dist)
            # Create covariance guess for each group
            for j in range(m):
                idx = np.where(groups == j)[1]
                hold = data_t[:, idx]
                if hold.shape[1] == 1:
                    cov[:, :, j] = np.identity(2)
                elif idx == []:
                    cov[:, :, j] = np.identity(2)
                else:
                    cov[:, :, j] = np.cov(hold, rowvar=True) + 1e-10*np.identity(2)
            # Singularity check
            for j in range(m):
                test = np.linalg.det(cov[:, :, j])
                if test == np.nan:
                    cov[:, :, j] = np.identity(2)
            # Begin EM portion of the problem, and run until converged
            converge = 20
            count = 0
            while converge >= 0.05 and count <= 500:
                px = np.zeros((m, n))
                for j in range(n):
                    for m_sub in range(m):
                        try:
                            px[m_sub, j] = alpha[0, m_sub] * evaluate_gaussian(data_t[:, j], mu[:, :, m_sub], cov[:, :, m_sub])
                        except IndexError:
                            px[m_sub, j] = alpha[m_sub] * evaluate_gaussian(data_t[:, j], mu[:, :, m_sub], cov[:, :, m_sub])
                # Find totals
                sum = np.sum(px, axis=0)
                p_lgx = np.divide(px, mlib.repmat(sum, m, 1))
                # Calculate weight matrix
                sum2 = np.zeros((m, 1))
                sum2[:, 0] = np.sum(p_lgx, axis=1)
                w = np.divide(p_lgx, mlib.repmat(sum2, 1, n))
                # Update alpha
                alpha_new = np.mean(p_lgx, axis=1)
                # Update mean
                mu_h = data_t.dot(w.T)
                # Reshape
                mu_new = np.zeros((1, 2, m))
                for j in range(m):
                    mu_new[:, :, j] = mu_h[:, j]
                # Update covariance
                cov_new = np.zeros((2, 2, m))
                for j in range(m):
                    v = np.subtract(data_t, mu_new[:, :, j].T)
                    u = np.multiply(mlib.repmat(w[j, :], 2, 1), v)
                    cov_new[:, :, j] = u.dot(v.T) + 1e-10*np.identity(2)
                # convergence check
                d_alpha = np.amax(np.subtract(alpha, alpha_new))
                d_mu = np.amax(np.subtract(mu, mu_new))
                d_cov = np.amax(np.subtract(cov, cov_new))
                converge = np.amax([d_alpha, d_mu, d_cov])
                # Update parameters and return to top
                alpha = alpha_new
                mu = mu_new
                cov = cov_new
                count += 1
            # Evaluate log-likelihood
            ll = np.zeros((m, n))
            for j in range(n):
                for m_sub in range(m):
                    try:
                        ll[m_sub, j] = alpha[0, m_sub] * evaluate_gaussian(data_t[:, j], mu[:, :, m_sub],
                                                                           cov[:, :, m_sub])
                    except IndexError:
                        ll[m_sub, j] = alpha[m_sub] * evaluate_gaussian(data_t[:, j], mu[:, :, m_sub],
                                                                        cov[:, :, m_sub])
            hold = np.sum(np.log(11))
            ll_mat[m-1, k] = np.sum(np.log(ll+1e-10)) / (-m*n)
    # alpha, mu, cov = output_validation(data_v, 4)
    scores = np.mean(ll_mat, axis=1)
    plt.plot(scores, 'bo-')
    plt.show()
    best_fit = np.where(scores == np.nanmax(scores))[0]

    return best_fit, data_v


def output_validation(data_v, m0):
    M = [m0[0, 0]]
    data_t = data_v
    for m in M:
        # Initialization
        mu = np.zeros((1, 2, m))
        cov = np.zeros((2, 2, m))
        n = data_t.shape[1]
        # Select m points to initialize parameters
        alpha = (1 / m) * np.ones((1, m))
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
        groups = np.zeros((1, n))
        for j in range(n):
            groups[0, j] = np.where(mins[j] == dist[:, j])[0][0]
        hold_g = np.where(mins == dist)
        # Create covariance guess for each group
        for j in range(m):
            idx = np.where(groups == j)[1]
            hold = data_t[:, idx]
            if hold.shape[1] == 1:
                cov[:, :, j] = np.identity(2)
            elif idx == []:
                cov[:, :, j] = np.identity(2)
            else:
                cov[:, :, j] = np.cov(hold, rowvar=True) + 1e-10 * np.identity(2)
        # Singularity check
        for j in range(m):
            test = np.linalg.det(cov[:, :, j])
            if test == np.nan:
                cov[:, :, j] = np.identity(2)
        # Begin EM portion of the problem, and run until converged
        converge = 20
        while converge >= 0.05:
            px = np.zeros((m, n))
            for j in range(n):
                for m_sub in range(m):
                    try:
                        px[m_sub, j] = alpha[0, m_sub] * evaluate_gaussian(data_t[:, j], mu[:, :, m_sub],
                                                                           cov[:, :, m_sub])
                    except IndexError:
                        px[m_sub, j] = alpha[m_sub] * evaluate_gaussian(data_t[:, j], mu[:, :, m_sub],
                                                                        cov[:, :, m_sub])
            # Find totals
            sum = np.sum(px, axis=0)
            p_lgx = np.divide(px, mlib.repmat(sum, m, 1))
            # Calculate weight matrix
            sum2 = np.zeros((m, 1))
            sum2[:, 0] = np.sum(p_lgx, axis=1)
            w = np.divide(p_lgx, mlib.repmat(sum2, 1, n))
            # Update alpha
            alpha_new = np.mean(p_lgx, axis=1)
            # Update mean
            mu_h = data_t.dot(w.T)
            # Reshape
            mu_new = np.zeros((1, 2, m))
            for j in range(m):
                mu_new[:, :, j] = mu_h[:, j]
            # Update covariance
            cov_new = np.zeros((2, 2, m))
            for j in range(m):
                v = np.subtract(data_t, mu_new[:, :, j].T)
                u = np.multiply(mlib.repmat(w[j, :], 2, 1), v)
                cov_new[:, :, j] = u.dot(v.T) + 1e-10 * np.identity(2)
            # convergence check
            d_alpha = np.amax(np.subtract(alpha, alpha_new))
            d_mu = np.amax(np.subtract(mu, mu_new))
            d_cov = np.amax(np.subtract(cov, cov_new))
            converge = np.amax([d_alpha, d_mu, d_cov])
            # Update parameters and return to top
            alpha = alpha_new
            mu = mu_new
            cov = cov_new
        # return

        return alpha, mu, cov


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
    samples = [100, 1000]
    # Call EM for GMM 100 times
    for N in samples:
        fit = []
        for runs in range(100):
            best_fit, data_v = em_gmm(alpha, mu, cov, N)
            fit.append(best_fit)
        m = scipy.stats.mode(np.asarray(fit))
        alpha, mu, cov = output_validation(data_v, m.mode)
        print(alpha, mu, cov)

    # Stop for results
