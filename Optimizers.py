import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        for param, grad in zip(params, gradients):
            param -= self.learning_rate * grad
