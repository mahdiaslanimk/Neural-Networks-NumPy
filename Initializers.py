import numpy as np


class LeCunUniform:
    def __init__(self):
        pass

    def init_params_for(self, shape):
        limit = np.sqrt(3 / float(shape[0]))
        return np.random.uniform(low=-limit, high=limit, size=shape)


class LeCunNormal:
    def __init__(self):
        pass

    def init_params_for(self, shape):
        limit = np.sqrt(1 / float(shape[0]))
        return np.random.normal(0.0, limit, size=shape)


class HeUniform:
    def __init__(self):
        pass

    def init_params_for(self, shape):
        limit = np.sqrt(6 / float(shape[0]))
        return np.random.uniform(low=-limit, high=limit, size=shape)


class HeNormal:
    def __init__(self):
        pass

    def init_params_for(self, shape):
        limit = np.sqrt(2 / float(shape[0]))
        return np.random.normal(0.0, limit, size=shape)


class XavierUniform:
    def __init__(self):
        pass

    def init_params_for(self, shape):
        if len(shape) == 1:
            limit = np.sqrt(6 / float(shape[0]))
        else:
            limit = np.sqrt(6 / float(shape[0] + shape[1]))
        return np.random.uniform(low=-limit, high=limit, size=shape)


class XavierNormal:
    def __init__(self):
        pass

    def init_params_for(self, shape):
        if len(shape) == 1:
            limit = np.sqrt(2 / float(shape[0]))
        else:
            limit = np.sqrt(2 / float(shape[0] + shape[1]))
        return np.random.normal(0.0, limit, size=shape)


initializers_dict = {
    "lecununiform": LeCunUniform(),
    "lecunnormal": LeCunNormal(),
    "xavieruniform": XavierUniform(),
    "xaviernormal": XavierNormal(),
    "heuniform": HeUniform(),
    "henormal": HeNormal(),
}


def initializer_set(name):
    name = name
    if type(name) == str:
        name = name.replace("_", "")
    else:
        raise Exception("Initializer's name should be a string")
    return initializers_dict[name]
