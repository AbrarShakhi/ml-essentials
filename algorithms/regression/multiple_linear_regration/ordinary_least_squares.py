import numpy as np

class OrdinaryLeastSquares:
    def __init__(self):
        self.coefficients_ = None

    @staticmethod
    def __add_intercept(X):
        # Add a column of ones to X for the intercept term
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        # Add intercept column
        X = self.__add_intercept(np.array(X))
        y = np.array(y)

        # Calculate the coefficients using OLS formula: (X.T X)^-1 X.T y
        X_transpose = X.T
        X_transpose_dot_X = X_transpose @ X
        X_transpose_dot_X_inverse = np.linalg.inv(X_transpose_dot_X)
        X_transpose_dot_y = X_transpose @ y

        self.coefficients_ = X_transpose_dot_X_inverse @ X_transpose_dot_y

        return self

    def predict(self, X):
        if self.coefficients_ is None:
            raise Exception("Model is not fitted.")

        X = self.__add_intercept(np.array(X))
        return X @ self.coefficients_
