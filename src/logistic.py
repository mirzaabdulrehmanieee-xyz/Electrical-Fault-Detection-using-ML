
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis(X, theta):
    return sigmoid(X @ theta)


def cost(X, y, theta):
    h = hypothesis(X, theta)
    h = np.clip(h, 1e-12, 1 - 1e-12)

    J = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J




def fit_logistic(X, y, alpha, n_iters):
    theta = np.zeros(X.shape[1])
    costs = []

    for i in range(n_iters):
        h = hypothesis(X, theta)
        gradient = (X.T @ (y - h)) / len(y)
        theta = theta + alpha * gradient
        costs.append(cost(X, y, theta))

    return theta, costs


def predict(X, theta):
    h = hypothesis(X, theta)
    return (h >= 0.5).astype(int)
