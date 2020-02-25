import numpy as np
import matplotlib.pyplot as plt
import math


def generate_gauss_2d_2sub(q, q_sub, m, c, samples):
    # Pull parameters from inputs
    n_labels = len(q)
    n_dim = m[0, :, 0].shape[0]
    try:
        sub = q_sub.shape[1]
    except IndexError:
        sub = 1
    # Random vector for determining class labels
    s = np.random.uniform(0, 1, (1, samples))
    d = np.zeros((n_dim, s.shape[1]))
    label_idx = np.zeros((1, s.shape[1]))
    for k in range(n_labels):
        prev = np.sum(q[0:k])
        idx = np.where((s[0, :] > prev) & (s[0, :] <= prev + q[k]))[0]
        label_idx[0, idx] = k
        s_sub = np.random.uniform(0, 1, (1, idx.shape[0]))
        if sub == 1:
            for j in idx:
                d[:, j] = np.random.multivariate_normal(m[0, :, k], c[:, :, k]).T
        else:
            for j in range(sub):
                prev = np.sum(q_sub[k, 0:j])
                idx_sub = np.where((s_sub[0, :] > prev) & (s_sub[0, :] <= prev + q_sub[k, j]))[0]
                for g in idx_sub:
                    d[:, idx[g]] = np.random.multivariate_normal(m[0, :, ((k*sub)+j)], c[:, :, ((k*sub)+j)]).T
                    ph = 1

    idx_0 = np.where(label_idx[0, :] == 0)[0]
    plt.plot(d[0, idx_0], d[1, idx_0], 'bo')
    idx_1 = np.where(label_idx[0, :] == 1)[0]
    plt.plot(d[0, idx_1], d[1, idx_1], 'k*')
    plt.show()

    return d, label_idx


def evaluate_gaussian(x, m, cov):
    l = (1/(np.sqrt((2*math.pi)**2*np.linalg.det(cov)))) * math.exp(-0.5*(np.dot(np.subtract(x, m).dot(np.linalg.inv(cov)), np.subtract(x, m))))

    return l


def plot_data(data, label_idx, D, m, c, q_sub, thresh):
    # Plot True Positives
    idx11 = np.where((label_idx[0, :] == 1) & (D[0, :] == 1))[0]
    # False positives
    idx10 = np.where((label_idx[0, :] == 0) & (D[0, :] == 1))[0]
    # True Negatives
    idx00 = np.where((label_idx[0, :] == 0) & (D[0, :] == 0))[0]
    # False Negatives
    idx01 = np.where((label_idx[0, :] == 1) & (D[0, :] == 0))[0]
    # Plot each with corresponding colors and symbols
    fig, axs = plt.subplots()
    axs.plot(data[0, idx11], data[1, idx11], 'ko', ms=6)
    axs.plot(data[0, idx10], data[1, idx10], 'rD', ms=6)
    axs.plot(data[0, idx00], data[1, idx00], 'bD', ms=6)
    axs.plot(data[0, idx01], data[1, idx01], 'ro', ms=6)
    axs.grid(True)
    [X, Y] = decision_boundaries(axs, m, c, q_sub, thresh)
    axs.plot(X, Y, 'g-', linewidth=5)
    axs.legend(['True Positives', 'False Positives', 'True Negatives', 'False Negatives', 'Decision Threshold'])

    return


def decision_boundaries(axs, m, c, q_sub, thresh):
    xbound = axs.get_xlim()
    ybound = axs.get_ylim()
    X = np.linspace(xbound[0]*0.5, xbound[1]*0.5, 20)
    Y = np.linspace(ybound[0]*0.5, ybound[1]*0.5, 20)
    [X, Y] = np.meshgrid(X, Y)
    # Calculate decision values and pick min from each row
    x = []
    for j in range(Y.shape[0]):
        sample = np.array([[X[j, :]], [Y[j, :]]])
        w = []
        for g in sample.T:
            k = g[0, :]
            w.append(np.log(q_sub[1] * evaluate_gaussian(k, m[0, :, 1], c[:, :, 1])) -
                       np.log(
                          q_sub[0] * evaluate_gaussian(k, m[0, :, 0], c[:, :, 0]))
                     )
        w = np.asarray(w)
        cent = np.abs(np.subtract(w, thresh))
        min = np.where(cent == np.amin(cent))[0]
        x.append(X[0, min])
    X = np.asarray(x)

    return X, Y


def normal_2d_2class(data, means, covariances):
    # Pull covariances
    cov0 = covariances[0:2, :]
    cov1 = covariances[2:4, :]
    # Pull means
    m0 = means[0]
    m1 = means[1]
    # List to track decisions
    wout = []
    for k in data.T:
        w = np.log(evaluate_gaussian(k, m1, cov1)) - np.log(evaluate_gaussian(k, m0, cov0))
        wout.append(w)
    w = np.asarray(wout)

    return w


def ROC_accuracy_check(data, label_idx, d):
    # Find percentage accuracy for decision 1
    mask_0 = np.where(label_idx[0, :] == 0)[0]
    mask_1 = np.where(label_idx[0, :] == 1)[0]
    # False alarms
    fa = len(np.where(d[0, mask_0] == 1)[0]) / len(mask_0)
    tp = len(np.where(d[0, mask_1] == 1)[0]) / len(mask_1)

    return fa, tp


def accuracy(label_idx, d):
    n_right = np.where(label_idx == d)[0]
    n_right_percent = len(n_right)/label_idx.shape[1]
    percent_error = 1 - n_right_percent

    return percent_error


def make_decision(w, gamma):
    d = np.zeros((1, len(w)))
    idx = np.where(w > gamma)[0]
    d[0, idx] = 1
    if gamma == 4:
        ph = 1

    return d


def gradient_descent(z, theta_initial, train_idx, alpha):
    eps = 10
    m = train_idx.shape[1]
    theta = theta_initial
    idx0 = np.where(train_idx[0, :] == 0)[0]
    idx1 = np.where(train_idx[0, :] == 1)[0]
    cost = []
    count = 0
    while eps >= 0.05 and count <= 2000:
        # Evaluate Sigmoid of current parameter
        h = 1 / (1 + np.exp(-1*theta.dot(z)))
        # Evaluate cost
        cost1 = np.sum(-1 * np.log(0.01 + h[0, idx1]))
        cost0 = np.sum(-1 * np.log(1.010 - h[0, idx0]))
        cost.append((-1/m) * (cost1 + cost0))
        # Find new theta from gradient
        theta_new = -np.dot(z, (h-train_idx).T).T
        # Check if converge
        eps = np.amax(np.abs(theta_new))
        # Update theta and run again
        theta = theta + alpha*theta_new
        count += 1
    cost = np.asarray(cost)

    return theta, cost


def draw_boundary_theta(theta, axs):
    xbound = axs.get_xlim()
    ybound = axs.get_ylim()
    X = np.linspace(xbound[0], xbound[1], 200)
    Y = np.linspace(ybound[0], ybound[1], 200)
    [X, Y] = np.meshgrid(X, Y)
    bias = np.ones((1, X.shape[1]))
    x = []
    for j in range(Y.shape[1]):
        ph = X[j, :]
        ph2 = Y[j, :]
        sample = np.row_stack((bias[0, :], X[j, :], Y[j, :]))
        test_vec = np.abs(theta.dot(sample))
        idx = np.where(test_vec[0, :] == np.amin(test_vec[0, :]))[0]
        x.append(X[j, idx])
    X = np.asarray(x)
    Y = Y[:, 0]

    return X, Y


def draw_boundary_theta_quad(theta, axs):
    xbound = axs.get_xlim()
    ybound = axs.get_ylim()
    X = np.linspace(xbound[0]*0.5, xbound[1]*0.5, 100)
    Y = np.linspace(ybound[0]*0.4, ybound[1]*0.4, 100)
    [X, Y] = np.meshgrid(X, Y)
    bias = np.ones((1, X.shape[1]))
    x = []
    for j in range(Y.shape[1]):
        ph = X[j, :]
        ph2 = Y[j, :]
        sample = np.row_stack((bias[0, :], X[j, :], Y[j, :], np.square(X[j, :]), np.multiply(X[j, :], Y[j, :]), np.square(Y[j, :])))
        test_vec = np.abs(theta.dot(sample))
        idx = np.where(test_vec[0, :] == np.amin(test_vec[0, :]))[0]
        x.append(X[j, idx])
    X = np.asarray(x)
    Y = Y[:, 0]

    return X, Y


def data_validation(data, theta, label_idx):
    m = data.shape[1]
    bias = np.ones((1, m))
    z = np.concatenate((bias, data))
    h = 1 / (1 + np.exp(-1 * theta.dot(z)))
    err = np.divide(np.sum(np.abs(np.subtract(label_idx, h))), m)

    return err


def data_validation_quad(data, theta, label_idx):
    m = data.shape[1]
    bias = np.ones((1, m))
    ph = np.multiply(data[0, :], data[1, :])
    z = np.row_stack((bias, data, np.square(data[0, :]), np.multiply(data[0, :], data[1, :]), np.square(data[1, :])))
    h = 1 / (1 + np.exp(-1 * theta.dot(z)))
    err = np.divide(np.sum(np.abs(np.subtract(label_idx, h))), m)

    return err


def question1_1():
    # Define distribution parameters
    q = np.array([0.9, 0.1])
    q_sub = np.array([1, 1])
    # Means
    m = np.zeros((1, 2, 2))
    m[0, :, 0] = [-2, 0]
    m[0, :, 1] = [2, 0]
    # Covariance
    c = np.zeros((2, 2, 2))
    c[:, :, 0] = np.array([[1, -0.9], [-0.9, 2]])
    c[:, :, 1] = np.array([[2, 0.9], [0.9, 1]])
    # Generate data
    data, label_idx = generate_gauss_2d_2sub(q, q_sub, m, c, 10000)
    dec = []
    for k in data.T:
        dec.append(np.log(q_sub[1] * evaluate_gaussian(k, m[0, :, 1], c[:, :, 1])) - (
                       np.log(
                          q_sub[0] * evaluate_gaussian(k, m[0, :, 0], c[:, :, 0]))
                   ))
    dec = np.asarray(dec)
    # Get min and max values
    bot = np.amin(dec)
    top = np.amax(dec)
    # Vary threshold and make ROC and error curves
    gamma = np.linspace(bot, top, 500)
    fa = []
    tp = []
    acc = []
    for g in gamma:
        D = make_decision(dec, g)
        fa_out, tp_out = ROC_accuracy_check(data, label_idx, D)
        fa.append(fa_out)
        tp.append(tp_out)
        acc.append(accuracy(label_idx, D))
    fa = np.asarray(fa)
    tp = np.asarray(tp)
    err = np.asarray(acc)
    # Data driven threshold minimum
    min_thresh = gamma[np.where(err == np.amin(err))[0][0]]
    # Evaluate at theoretical best threshold
    D = make_decision(dec, min_thresh)
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, D)
    err_out = accuracy(label_idx, D)
    print(err_out)
    print(min_thresh)
    # Plots
    fig, axs = plt.subplots(1, 2)
    # ROC Plot
    axs[0].plot(fa, tp, linewidth=3)
    axs[0].axvline(x=fa_out, color='b')
    axs[0].annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out + 0.01, 0.4), xycoords='data')
    axs[0].axhline(y=tp_out, color='r')
    axs[0].annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.4, tp_out + 0.01), xycoords='data')
    axs[0].set_title('ROC Curve for Problem 2')
    axs[0].set_xlabel('P(false alarm)')
    axs[0].set_ylabel('P(hit)')
    # Error Plot
    axs[1].plot(gamma, err)
    axs[1].set_title('Error versus threshold')
    axs[1].set_xlabel('Threshold Value')
    axs[1].set_ylabel('P(Error)')
    plot_data(data, label_idx, D, m, c, q_sub, min_thresh)
    # plt.show()

    return data, label_idx


def question1_2(data, label_idx):
    # This part will use linear logistic regression to classify the data
    # Initialization
    theta_initial = np.zeros((1, 3))
    # vector of training values
    train = [10, 100, 1000]
    # Optimize
    for k in train:
        train = np.zeros((2, k))
        train_idx = np.zeros((1, k))
        for j in range(k):
            # Sample training data with replacement
            rand = np.random.randint(0, data.shape[1])
            train[:, j] = data[:, rand]
            train_idx[0, j] = label_idx[0, rand]
        m = train.shape[1]
        bias = np.ones((1, m))
        z = np.concatenate((bias, train))
        [theta, cost] = gradient_descent(z, theta_initial, train_idx, 0.005)
        fig, axs = plt.subplots(1, 2)
        idx0 = np.where(label_idx[0, :] == 0)[0]
        idx1 = np.where(label_idx[0, :] == 1)[0]
        axs[0].plot(data[0, idx0], data[1, idx0], 'bo')
        axs[0].plot(data[0, idx1], data[1, idx1], 'k*')
        # Plot decision boundary
        X, Y = draw_boundary_theta(theta, axs[0])
        axs[0].plot(X, Y, 'g-', linewidth=3)
        axs[1].plot(cost)
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Cost')
        err = data_validation(data, theta, label_idx)
        print(err)


def question1_3(data, label_idx):
    # Init
    theta_initial = np.zeros((1, 6))
    train = [10, 100, 1000, 10000]
    for k in train:
        train = np.zeros((2, k))
        train_idx = np.zeros((1, k))
        squ = np.zeros((2, k))
        cross = np.zeros((1, k))
        for j in range(k):
            # Sample training data with replacement
            rand = np.random.randint(0, data.shape[1])
            # X1 and X2
            train[:, j] = data[:, rand]
            # X1^2
            squ[:, j] = np.square(data[:, rand])
            cross[0, j] = np.multiply(data[0, rand], data[1, rand])
            train_idx[0, j] = label_idx[0, rand]
        m = train.shape[1]
        bias = np.ones((1, m))
        z = np.row_stack((bias, train, squ[0, :], cross, squ[1, :]))
        [theta, cost] = gradient_descent(z, theta_initial, train_idx, 0.005)
        fig, axs = plt.subplots(1, 2)
        idx0 = np.where(label_idx[0, :] == 0)[0]
        idx1 = np.where(label_idx[0, :] == 1)[0]
        axs[0].plot(data[0, idx0], data[1, idx0], 'bo')
        axs[0].plot(data[0, idx1], data[1, idx1], 'k*')
        # Plot decision boundary
        X, Y = draw_boundary_theta_quad(theta, axs[0])
        axs[0].plot(X, Y, 'g-', linewidth=3)
        axs[1].plot(cost)
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Cost')
        err = data_validation_quad(data, theta, label_idx)
        print(err)


if __name__ == '__main__':
    data, label_idx = question1_1()
    question1_2(data, label_idx)
    question1_3(data, label_idx)
    plt.show()