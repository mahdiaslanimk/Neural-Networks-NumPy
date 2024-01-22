import numpy as np


class MeanSquaredError:
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def compute_derivative(self, y_true, y_pred):
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)
