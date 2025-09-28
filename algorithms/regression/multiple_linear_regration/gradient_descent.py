import numpy as np


class GradientDescent:
    def __init__(self, learning_rate: float=0.01, iterations: int=1000):
        self.learning_rate: float = learning_rate
        self.iterations: int = iterations
        self.theta = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m = len(y)

        X = np.c_[np.ones(m), X]

        self.theta = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            predictions = np.dot(X, self.theta)
            gradient = (1/m) * np.dot(X.T, (predictions - y))
            self.theta -= self.learning_rate * gradient

        return self

    def predict(self, X):
        if self.theta is None:
            raise Exception("Model is not fitted.")

        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.theta)
