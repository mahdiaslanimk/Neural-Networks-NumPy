import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        for param, grad in zip(params, gradients):
            param -= self.learning_rate * grad


class MomentumSGD:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        for i in range(len(params)):
            self.velocity[i] = (
                self.momentum * self.velocity[i] + (1 - self.momentum) * gradients[i]
            )
            params[i] -= self.learning_rate * self.velocity[i]
