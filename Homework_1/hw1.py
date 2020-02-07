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
    # plt.plot(data_points[0, :], data_points[1, :])
    # plt.show()

    data = data_points

    return data, label_idx


def generate_gauss_2d_2sub(q, q_sub, m, c, samples):
    # Pull parameters from inputs
    n_labels = len(q)
    n_dim = m[0, :, 0].shape
    # Random vector for determining class labels
    s = np.random.uniform(0, 1, (1, samples))
    label_idx = np.zeros((1, s.shape[1]))
    for k in range(n_labels):
        if k == 0:
            idx = np.where(s[0, :] <= q[0])

        else:
            prev = np.sum(q[0:k])
            idx = np.where((s[0, :] > prev) & (s[0, :] <= prev + q[k]))[0]
            label_idx[0, idx] = k
    ph = 1


def evaluate_gaussian(x, m, cov):
    l = (1/(np.sqrt((2*math.pi)**2*np.linalg.det(cov)))) * math.exp(-0.5*(np.dot(np.subtract(x, m).dot(np.linalg.inv(cov)), np.subtract(x, m))))

    return l

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
    fa = np.asarray(fa)
    tp = np.asarray(tp)
    fig, ax = plt.subplots()
    ax.plot(fa, tp, linewidth=5)
    # Minimum error is at Gamma = 4.0
    d = make_decision(w, 4)
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, d)
    # Calculate error at this threshold
    percent_error = accuracy(label_idx, d)
    # Plot dotted lines on ROC curve for this point
    ax.axvline(x=fa_out, color='b')
    ax.annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out+0.01, 0.2), xycoords='data')
    ax.axhline(y=tp_out, color='r')
    ax.annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.02, tp_out+0.01), xycoords='data')
    ax.annotate('P(error): {:.3f}'.format(percent_error), xy=(0.6, 0.75), xycoords='figure fraction')
    ax.set_title('ROC for 1.1 Threshold: 4.0')
    ax.set_xlabel('P(False Alarm)')
    ax.set_ylabel('P(Hit)')

    return data, label_idx


def problem1_2(data, label_idx):
    # Use most of the same code from part 1, but after generating the data we will switch the covariance to a guess of I
    # Format mixture parameters
    means = np.array([[-0.1, 0], [0.1, 0]])
    # Part 2 wants us to vary gamma from 0 to inf
    fa = []
    tp = []
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
    fa = np.asarray(fa)
    tp = np.asarray(tp)
    fig, ax = plt.subplots()
    ax.plot(fa, tp, linewidth=5)
    # Minimum error is at Gamma = 4.0
    d = make_decision(w, 4)
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, d)
    # Calculate error at this threshold
    percent_error = accuracy(label_idx, d)
    # Plot dotted lines on ROC curve for this point
    ax.axvline(x=fa_out, color='b')
    ax.annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out + 0.01, 0.4), xycoords='data')
    ax.axhline(y=tp_out, color='r')
    ax.annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.2, tp_out + 0.01), xycoords='data')
    ax.annotate('P(error): {:.3f}'.format(percent_error), xy=(0.5, 0.75), xycoords='figure fraction')
    ax.set_title('ROC for 1.2 Threshold: 4.0')
    ax.set_xlabel('P(False Alarm)')
    ax.set_ylabel('P(Hit)')


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
    fig, ax = plt.subplots()
    ax.plot(gamma, acc)
    D = make_decision(d.T, gamma[min])
    print(gamma[min])
    fa_out, tp_out = ROC_accuracy_check(data, label_idx, D)
    fig, ax = plt.subplots()
    # Plot ROC curve
    ax.plot(fa, tp, linewidth=5)
    # Plot minimum error values
    # Calculate error at this threshold
    percent_error = accuracy(label_idx, D)
    # Plot dotted lines on ROC curve for this point
    ax.axvline(x=fa_out, color='b')
    ax.annotate('P(False Alarm): {:.3f}'.format(fa_out), xy=(fa_out + 0.01, 0.4), xycoords='data')
    ax.axhline(y=tp_out, color='r')
    ax.annotate('P(Hit): {:.3f}'.format(tp_out), xy=(0.4, tp_out + 0.01), xycoords='data')
    ax.annotate('P(error): {:.3f}'.format(percent_error), xy=(0.6, 0.6), xycoords='figure fraction')
    ax.set_xlabel('P(False Alarm)')
    ax.set_ylabel('P(Hit)')
    ax.set_title('ROC for 1.3 Threshold: 4.0')
    plt.show()


def problem2():
    # Initialize all mixture parameters
    # External and internal priors
    q = [0.9, 0.1]
    q_sub = np.array([[0.5, 0.5], [0.6, 0.4]])
    # means
    m = np.zeros((1, 2, 4))
    m[0, :, 0] = [1, 2]
    m[0, :, 1] = [1, 0]
    m[0, :, 2] = [-1, -1]
    m[0, :, 3] = [0, 1]
    # Covariances
    c = np.zeros((2, 2, 4))
    c[:, :, 0] = np.array([[1, 0.75], [0.75, 1]])
    c[:, :, 1] = np.array([[1, -0.2], [-0.2, 1]])
    c[:, :, 2] = np.array([[1, -0.4], [-0.4, 1]])
    c[:, :, 3] = np.array([[1, -0.1], [-0.1, 1]])
    # Call function to generate data
    data = generate_gauss_2d_2sub(q, q_sub, m, c, 10000)




if __name__ == '__main__':
    # problem 1_1 decision theory with known covariance
    data, label_idx = problem1_1()
    # problem 1_2 decision theory with Naive Bayes
    problem1_2(data, label_idx)
    # Problem 1_3 decision theory with Fischer LDA
    problem1_3(data, label_idx)
    # Problem 2 decision problem with two classes, each composed of two Gaussians
    problem2()
