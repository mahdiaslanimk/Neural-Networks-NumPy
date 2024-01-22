import numpy as np


class Accuracy:
    def __init__(self, name="accuracy", one_hot=False):
        self.name = name
        self.one_hot = one_hot

    def compute(self, y_true, y_pred):
        if self.one_hot:
            y_pred_labels = np.argmax(y_pred, axis=-1)  # Convert one-hot to labels
            correct_predictions = np.sum(y_true == y_pred_labels)
        total_samples = len(y_true)
        return correct_predictions / total_samples


class MeanAbsoluteError:
    def __init__(self, name="mae"):
        self.name = name

    def compute(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
