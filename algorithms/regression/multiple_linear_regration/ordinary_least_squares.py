import numpy as np


class OrdinaryLeastSquares:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, y_actual, y_pred):
        y_actual = np.array(y_actual)
        y_pred = np.array(y_pred)
        ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
        ss_res = np.sum((y_actual - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_total)
        return r2
