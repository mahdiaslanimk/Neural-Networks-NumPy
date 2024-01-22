
import numpy as np




def linear(x):
    return x

def linear_d(x):
    if type(x) == np.ndarray:
        return np.ones_like(x)
    else:
        return 1



def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return np.where(x > 0, 1, 0)



def unitstep(x):
    return np.sign(x)

def unitstep_d(x):
    return 0



def unitstep01(x):
    return np.where(x >= 0, 1, 0)

def unitstep01_d(x):
    return 0



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)



def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1 - np.tanh(x)**2



def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_d(x):
    S = softmax(x)
    dS_dx = S * (1 - S)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if i != j:
                dS_dx[i, j] = -S[i, j] * S[i, j]
    return dS_dx