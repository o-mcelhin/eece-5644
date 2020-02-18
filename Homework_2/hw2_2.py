import numpy as np
import matplotlib.pyplot as plt
import math


def data_generation(true_params, gamma, std_v, samples):
    # Generate x data from uniform [-1, 1]
    x = np.random.uniform(-1, 1, (1, samples))
    # Generate v as additive noise from N(0,std_v)
    v = np.random.normal(0, std_v, (1, samples))
    # Generate y values
    y = (true_params[0] * np.power(x, 3) + true_params[1] * np.power(x, 2) + true_params[2] * np.power(x, 1) +
         true_params[3] + v)
    # Check data
    # t = np.linspace(-1, 1, 500)
    # y_true = (true_params[0] * np.power(t, 3) + true_params[1] * np.power(t, 2) + true_params[2] * np.power(t, 1) +
    #           true_params[3])
    # fig, axs = plt.subplots()
    # axs.plot(t, y_true)
    # axs.plot(x, y, 'ro')
    # plt.show()
    # ph = 1

    return x, y


def MAP_regression(data_input, data_output, std_v, gamma):
    sum1_terms = np.zeros((4, 4, data_input.shape[1]))
    sum2_terms = np.zeros((1, 4, data_input.shape[1]))
    for j in range(data_input.shape[1]):
        sum1_terms[:, :, j] = np.outer(data_input[:, j], data_input[:, j].T)
        sum2_terms[:, :, j] = np.outer(data_output[0, j], data_input[:, j])
    sum1 = np.sum(sum1_terms, 2)
    sum2 = np.sum(sum2_terms, 2)
    t12 = (std_v**2 / gamma) * np.identity(4)

    theta = np.dot(np.linalg.inv(sum1 + t12), sum2.T)

    return theta


if __name__ == '__main__':
    w_true = [-0.5, 0.23, 0.9, -0.4]
    gamma = np.arange(-10, 10, 0.1)
    gamma = np.power(10, gamma)
    std_v = 0.05
    outputs = []
    for g in gamma:
        # Generate random w and x, y data
        w_gen = w_true
        theta = []
        for k in range(100):
            x, y = data_generation(w_gen, g, std_v, 10)
            # Bias x to get z
            bias = np.ones((1, x.shape[1]))
            z = np.row_stack([np.power(x, 3), np.power(x, 2), x, bias])
            # Initialize
            theta_out = MAP_regression(z, y, std_v, g)
            theta.append(theta_out)
        theta = np.asarray(theta)
        theta = theta[:, :, 0]
        diff = np.linalg.norm(np.subtract(w_true, theta), axis=1)
        diff_s = np.sort(diff)
        outputs.append([diff_s[0], diff_s[24], diff_s[49], diff_s[74], diff_s[99]])
    outputs = np.asarray(outputs)
    plt.semilogx(gamma, outputs[:, 0])
    plt.semilogx(gamma, outputs[:, 4])
    plt.show()
