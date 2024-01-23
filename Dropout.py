import numpy as np
from Functions import *
from Initializers import initializer_set


class Dropout:
    def __init__(
        self,
        units,
        input_dim,
        activation="linear",
        initialization="xavieruniform",
        dropout_rate=0.5,
    ):
        self.initializer = initializer_set(initialization)
        self.activation = eval(activation.lower())
        self.activation_d = eval((activation + "_d").lower())
        self.dropout_rate = dropout_rate
        self.w = self.initializer.init_params_for((units, input_dim))
        self.b = self.initializer.init_params_for((1, units))
        self.s = None
        self.f = None
        self.df = None
        self.g_w = None
        self.g_b = None

    def get_params(self):
        return [self.w, self.b]

    def get_gradients(self):
        g_w = np.dot(self.s.T, self.a_in)
        g_b = np.sum(self.s, axis=0, keepdims=True)
        return [g_w, g_b]

    def forward(self, a_in, training=False):
        self.a_in = a_in
        n = np.dot(self.a_in, self.w.T) + self.b
        self.f = self.activation(n)
        if training:
            self.mask = (
                np.random.rand(*self.f.shape) < (1 - self.dropout_rate)
            ).astype(int)
            self.f *= self.mask
        self.df = self.activation_d(n)
        return self.f

    def backward(self):
        self.s *= self.mask
        return np.dot(self.s, self.w)
