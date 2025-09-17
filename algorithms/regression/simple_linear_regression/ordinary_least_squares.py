import numpy as np


class OrdinaryLeastSquares:
    def __init__(self):
        self.coef_ = None  # slope
        self.intercept_ = None  # intercept

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        
        # Add bias term (for intercept)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope (coef_)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        self.coef_ = numerator / denominator
        
        # Calculate intercept
        self.intercept_ = y_mean - self.coef_ * x_mean

    def predict(self, x):
        x = np.array(x)
        return self.intercept_ + self.coef_ * x

    def score(self, y_actual, y_pred):
        y_actual = np.array(y_actual)
        y_pred = np.array(y_pred)
        ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
        ss_res = np.sum((y_actual - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_total)
        return r2
