import numpy as np
import math
import matplotlib.pyplot as plt


def data_gen_normal(samples, dim, n_labels, n_mix, label_priors, mixture_priors, means, covariances):
    sample_gen = np.random.uniform(0, 1, (1, samples))
    label_idx = np.zeros((1, samples))
    data_points = np.zeros((2, samples))
    # For each label, find how many points are in said class and set label_idx to hold the truth values
    for k in range(n_labels):
        if k == 0:
            idx = np.where((0 <= sample_gen[0, :]) & (sample_gen[0, :] < label_priors[0]))
            label_idx[0, idx] = k
            # Generate data points for it
            for j in idx[0]:
                data_points[:, j] = np.random.multivariate_normal(means[0, :], covariances[0:dim, :])
            # data_points.append(np.random.multivariate_normal(means[0, :], covariances[0:dim, :], size=len(idx[0])))
            # Remove the parameters from top of array
            means = np.delete(means, 0, axis=0)
            covariances = np.delete(covariances, range(dim), axis=0)
            # Plot them
        else:
            prev = sum(label_priors[:k])
            idx = np.where((prev <= sample_gen[0, :]) & (sample_gen[0, :] < prev+label_priors[k]))
            label_idx[0, idx] = k
            # Generate data points for it
            for j in idx[0]:
                data_points[:, j] = np.random.multivariate_normal(means[0, :], covariances[0:dim, :])
            # Remove the parameters from top of array
            means = np.delete(means, 0, axis=0)
            covariances = np.delete(covariances, range(dim), axis=0)
    fig, ax = plt.subplots()
    idx1 = np.where(label_idx[0, :] == 1)[0]
    idx0 = np.where(label_idx[0, :] == 0)[0]
    ax.plot(data_points[0, idx1], data_points[1, idx1], 'ko')
    ax.plot(data_points[0, idx0], data_points[1, idx0], 'bd')
    ax.legend(['Class 1', 'Class 0'])
    data = data_points

    return data, label_idx


def generate_gauss_2d_2sub(q, q_sub, m, c, samples):
    # Pull parameters from inputs
    n_labels = len(q)
    n_dim = m[0, :, 0].shape[0]
    sub = q_sub.shape[1]
    # Random vector for determining class labels
    s = np.random.uniform(0, 1, (1, samples))
    d = np.zeros((n_dim, s.shape[1]))
    label_idx = np.zeros((1, s.shape[1]))
    for k in range(n_labels):
        prev = np.sum(q[0:k])
        idx = np.where((s[0, :] > prev) & (s[0, :] <= prev + q[k]))[0]
        label_idx[0, idx] = k
        s_sub = np.random.uniform(0, 1, (1,idx.shape[0]))
        for j in range(sub):
            prev = np.sum(q_sub[k, 0:j])
            idx_sub = np.where((s_sub[0, :] > prev) & (s_sub[0, :] <= prev + q_sub[k, j]))[0]
            for g in idx_sub:
                d[:, idx[g]] = np.random.multivariate_normal(m[0, :, ((k*sub)+j)], c[:, :, ((k*sub)+j)]).T
                ph = 1

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


def decision_boundaries(axs, m, c, q_sub, thresh):
    xbound = axs.get_xlim()
    ybound = axs.get_ylim()
    X = np.linspace(xbound[0], xbound[1], 50)
    Y = np.linspace(ybound[0], ybound[1], 50)
    [X, Y] = np.meshgrid(X, Y)
    # Calculate decision values and pick min from each row
    x = []
    for j in range(Y.shape[0]):
        sample = np.array([[X[j, :]], [Y[j, :]]])
        w = []
        for g in sample.T:
            k = g[0, :]
            w.append(np.log(
                q_sub[1, 0] * evaluate_gaussian(k, m[0, :, 3], c[:, :, 3]) + q_sub[1, 1] * evaluate_gaussian(k,
                                                                                                             m[0, :, 2],
                                                                                                             c[:, :,
                                                                                                             2])) - (
                           np.log(q_sub[0, 0] * evaluate_gaussian(k, m[0, :, 1], c[:, :, 1]) + q_sub[
                               0, 1] * evaluate_gaussian(k, m[0, :, 0], c[:, :, 0]))
                       ))
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


def make_decision(w, gamma):
    d = np.zeros((1, len(w)))
    idx = np.where(w > gamma)[0]
    d[0, idx] = 1
    if gamma == 4:
        ph = 1

    return d


def accuracy(label_idx, d):
    n_right = np.where(label_idx == d)[0]
    n_right_percent = len(n_right)/label_idx.shape[1]
    percent_error = 1 - n_right_percent

    return percent_error


def problem1_1():
    # Format mixture parameters
    means = np.array([[-0.1, 0], [0.1, 0]])
    covariances = np.array([[1, -0.9], [-0.9, 1], [1, 0.9], [0.9, 1]])
    data, label_idx = data_gen_normal(10000, 2, 2, 1, [0.8, 0.2], [1, 1], means, covariances)
    # Part 2 wants us to vary gamma from 0 to inf
    fa = []
    tp = []
    acc = []
    # Calculate decision statistics
    w = normal_2d_2class(data, means, covariances)
    max = np.amax(w)
    min = np.amin(w)
    gamma = np.linspace(min-1, max, 500)
    for g in gamma:
        d = make_decision(w, g)
        fa_out, tp_out = ROC_accuracy_check(data, label_idx, d)
        fa.append(fa_out)
        tp.append(tp_out)
        acc.append(accuracy(label_idx, d))
    fa = np.asarray(fa)
    tp = np.asarray(tp)
    acc = np.asarray(acc)
    # Find minimum threshold value
    min_thresh = gamma[np.where(acc == np.amin(acc))[0]]
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(fa, tp, linewidth=5)
    # Minimum error is at Gamma = 4.0
    d = make_decision(w, min_thresh)
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, d)
    # Calculate error at this threshold
    percent_error = accuracy(label_idx, d)
    print(percent_error)
    print(min_thresh)
    # Plot dotted lines on ROC curve for this point
    ax[0].axvline(x=fa_out, color='b')
    ax[0].annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out+0.01, 0.2), xycoords='data')
    ax[0].axhline(y=tp_out, color='r')
    ax[0].annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.05, tp_out+0.01), xycoords='data')
    ax[0].annotate('P(error): {:.3f}'.format(percent_error), xy=(0.8, 0.4), xycoords='data')
    ax[0].set_title('ROC for 1.1 Threshold: 4.0')
    ax[0].set_xlabel('P(False Alarm)')
    ax[0].set_ylabel('P(Hit)')
    # Plot error as a function of threshold
    ax[1].plot(gamma, acc)
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('P(Error)')

    return data, label_idx


def problem1_2(data, label_idx):
    # Use most of the same code from part 1, but after generating the data we will switch the covariance to a guess of I
    # Format mixture parameters
    means = np.array([[-0.1, 0], [0.1, 0]])
    # Part 2 wants us to vary gamma from 0 to inf
    fa = []
    tp = []
    acc = []
    # Switch covariance estimates
    covariances = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    # Calculate decision statistics
    w = normal_2d_2class(data, means, covariances)
    max = np.amax(w)
    gamma = np.linspace(0, max, 500)
    for g in gamma:
        d = make_decision(w, g)
        fa_out, tp_out = ROC_accuracy_check(data, label_idx, d)
        fa.append(fa_out)
        tp.append(tp_out)
        acc.append(accuracy(label_idx, d))
    fa = np.asarray(fa)
    tp = np.asarray(tp)
    acc = np.asarray(acc)
    min_thresh = gamma[np.where(acc == np.amin(acc))[0][0]]
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(fa, tp, linewidth=5)
    # Minimum error is at Gamma = 4.0
    d = make_decision(w, min_thresh)
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, d)
    # Calculate error at this threshold
    percent_error = accuracy(label_idx, d)
    print(percent_error)
    print(min_thresh)
    # Plot dotted lines on ROC curve for this point
    ax[0].axvline(x=fa_out, color='b')
    ax[0].annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out + 0.01, 0.4), xycoords='data')
    ax[0].axhline(y=tp_out, color='r')
    ax[0].annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.2, tp_out + 0.01), xycoords='data')
    ax[0].annotate('P(error): {:.3f}'.format(percent_error), xy=(0.3, 0.2), xycoords='data')
    ax[0].set_title('ROC for 1.2 Threshold: 4.0')
    ax[0].set_xlabel('P(False Alarm)')
    ax[0].set_ylabel('P(Hit)')
    # Plot error curve
    ax[1].plot(gamma, acc)
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('P(Error)')

def problem1_3(data, label_idx):
    # Get sample mean and variance for each class
    idx0 = np.where(label_idx[0, :] == 0)[0]
    data0 = data[:, idx0]
    # Calculate from samples
    m0 = np.mean(data0, axis=1)
    cov0 = np.cov(data0, rowvar=True)
    # Same for class 1
    idx1 = np.where(label_idx[0, :] == 1)[0]
    data1 = data[:, idx1]
    # Calculate from samples
    m1 = np.mean(data1, axis=1)
    cov1 = np.cov(data1, rowvar=True)
    # Solve for W
    Sw_inv = np.linalg.inv(np.subtract(cov0, cov1))
    Sb = np.subtract(m0, m1).dot(np.subtract(m0, m1).T)
    W_mat = Sw_inv.dot(Sb)
    [l, v] = np.linalg.eig(W_mat)
    w = v[:, np.where(l == np.amax(l))[0]]
    w = w.T
    # Calculate decision scores
    d = w.dot(data)
    # Bounds for threshold
    min = np.amin(d)
    max = np.amax(d)
    # Create ROC
    fa = []
    tp = []
    acc = []
    gamma = np.linspace(min, max, 500)
    for g in gamma:
        D = make_decision(d.T, g)
        fa_out, tp_out = ROC_accuracy_check(data, label_idx, D)
        fa.append(fa_out)
        tp.append(tp_out)
        acc.append(accuracy(label_idx, D))
    fa = np.asarray(fa)
    tp = np.asarray(tp)
    acc = np.asarray(acc)
    # find min error
    min = np.where(acc == np.amin(acc))[0]
    # Solve at Gamma for min error
    D = make_decision(d.T, gamma[min])
    print(gamma[min])
    print(np.amin(acc))
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, D)
    fig, ax = plt.subplots(1, 2)
    # Plot ROC curve
    ax[0].plot(fa, tp, linewidth=5)
    # Plot minimum error values
    # Calculate error at this threshold
    percent_error = accuracy(label_idx, D)
    # Plot dotted lines on ROC curve for this point
    ax[0].axvline(x=fa_out, color='b')
    ax[0].annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out + 0.01, 0.4), xycoords='data')
    ax[0].axhline(y=tp_out, color='r')
    ax[0].annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.4, tp_out + 0.01), xycoords='data')
    ax[0].annotate('P(error): {:.3f}'.format(percent_error), xy=(0.8, 0.4), xycoords='data')
    ax[0].set_xlabel('P(False Alarm)')
    ax[0].set_ylabel('P(Hit)')
    ax[0].set_title('ROC for 1.3 Threshold: 4.0')
    # Error curve
    ax[1].plot(gamma, acc)
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('P(Error)')




def problem2():
    # Initialize all mixture parameters
    # External and internal priors
    q = [0.6, 0.4]
    q_sub = np.array([[0.7, 0.3], [0.6, 0.4]])
    # means
    m = np.zeros((1, 2, 4))
    m[0, :, 0] = [2, 1]
    m[0, :, 1] = [1, -1]
    m[0, :, 2] = [-1, -1]
    m[0, :, 3] = [0, 2]
    # Covariances
    c = np.zeros((2, 2, 4))
    c[:, :, 0] = np.array([[1, 0.25], [0.25, 1]])
    c[:, :, 1] = np.array([[1, -0.2], [-0.2, 1]])
    c[:, :, 2] = np.array([[1, -0.4], [-0.4, 1]])
    c[:, :, 3] = np.array([[1, 0.4], [0.4, 1]])
    # Call function to generate data
    data, label_idx = generate_gauss_2d_2sub(q, q_sub, m, c, 1000)
    # Calculate decision statistics
    dec = []
    for k in data.T:
        dec.append(np.log(q_sub[1, 0]*evaluate_gaussian(k, m[0, :, 3], c[:, :, 3]) + q_sub[1, 1]*evaluate_gaussian(k, m[0, :, 2], c[:, :, 2])) - (
            np.log(q_sub[0, 0]*evaluate_gaussian(k, m[0, :, 1], c[:, :, 1]) + q_sub[0, 1]*evaluate_gaussian(k, m[0, :, 0], c[:, :, 0]))
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


def problem3():
    m0 = -2
    m1 = 2
    var = 1
    # Index
    x = np.linspace(-5, 5, 500)
    # Distributions
    dist0 = (1/math.sqrt(2*math.pi)) * np.exp(np.multiply(-0.5, np.power(np.divide((x - m0), var), 2)))
    dist1 = (1/math.sqrt(2*math.pi)) * np.exp(-0.5 * np.power((np.divide((x - m1), var)), 2))
    fig, axs = plt.subplots()
    axs.plot(x, dist0, 'b-')
    axs.plot(x, dist1, 'k-')
    axs.legend(['Distribution 0', 'Distribution 1'])


if __name__ == '__main__':
    # problem 1_1 decision theory with known covariance
    data, label_idx = problem1_1()
    # problem 1_2 decision theory with Naive Bayes
    problem1_2(data, label_idx)
    # Problem 1_3 decision theory with Fischer LDA
    problem1_3(data, label_idx)
    # Problem 2 decision problem with two classes, each composed of two Gaussians
    problem2()
    # Problem 3: plots and other stuff
    problem3()
    plt.show()
