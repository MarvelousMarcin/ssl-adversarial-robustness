import numpy as np


def linear_CKA(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord='fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord='fro') ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))
