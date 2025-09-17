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
        return self

    def predict(self, x):
        if self.intercept_ is None or self.coef_ is None:
            raise Exception("Model is not fitted.")

        x = np.array(x)
        return self.intercept_ + self.coef_ * x
