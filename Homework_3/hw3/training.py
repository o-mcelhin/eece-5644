from sklearn.neural_network import MLPClassifier
import numpy as np


def train_MLP(input, output, layers=(5), f_activation='relu'):
    # Initialize a model
    model = MLPClassifier(hidden_layer_sizes=layers, activation=f_activation, verbose=False, max_iter=1000)
    # fit the model
    model.fit(input, output)

    return model


def make_decision(p):
    top = np.amax(p, axis=1)
    dec = [np.where(p[j, :] == top[j])[0][0] for j in range(p.shape[0])]

    return dec


def validate(dec, labels_t):
    sub = np.subtract(dec, labels_t)
    mask = sub == 0
    right = np.sum(mask, axis=1)
    pct = right/labels_t.shape[1]

    return pct[0]
