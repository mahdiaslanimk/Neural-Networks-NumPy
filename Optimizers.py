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


class VLBP:
    def __init__(
        self,
        model,
        x_train,
        y_train,
        learning_rate,
        momentum=None,
        xi=0.01,
        rho=0.9,
        eta=1.2,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.xi = xi
        self.rho = rho
        self.eta = eta
        self.optimizer = (
            MomentumSGD(learning_rate=self.learning_rate, momentum=self.momentum)
            if momentum
            else SGD(learning_rate=self.learning_rate)
        )
        self.model = model
        self.prev_squared_error = -np.inf
        self.x_train = x_train
        self.y_train = y_train

    def update(self, param, gradients):
        self.cloned_model = self.model.clone_model()
        self.cloned_model.update_params(self.optimizer)
        squared_error = np.mean(
            np.square(self.y_train - self.cloned_model.forward(self.x_train))
        )
        self.update_learning_rate(squared_error)

    def update_learning_rate(self, squared_error):
        # VLBP rules
        if squared_error > (1 + self.xi) * self.prev_squared_error:
            # Rule 1: Discard weight update
            self.learning_rate *= self.rho
            if self.momentum != None:  # if using momentum
                self.momentum = 0
        elif squared_error < self.prev_squared_error:
            # Rule 2: Accept weight update and increase learning rate
            self.accept_update()
            self.learning_rate *= self.eta
            if self.momentum == 0:
                self.momentum = self.momentum
        else:
            # Rule 3: Accept weight update but keep learning rate unchanged
            self.accept_update()
            if self.momentum == 0:
                self.momentum = self.momentum

        self.prev_squared_error = squared_error

    def accept_update(self):
        for i, cloned_layer in enumerate(self.cloned_model.layers):
            self.model.layers[i] = cloned_layer


class RMSprop:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.accumulated_squared_gradients = None

    def update(self, params, gradients):
        if self.accumulated_squared_gradients is None:
            self.accumulated_squared_gradients = [
                np.zeros_like(param) for param in params
            ]

        for i in range(len(params)):
            self.accumulated_squared_gradients[
                i
            ] = self.beta * self.accumulated_squared_gradients[i] + (1 - self.beta) * (
                gradients[i] ** 2
            )

            params[i] -= (
                self.learning_rate
                * gradients[i]
                / (np.sqrt(self.accumulated_squared_gradients[i]) + self.epsilon)
            )


class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, gradients):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
