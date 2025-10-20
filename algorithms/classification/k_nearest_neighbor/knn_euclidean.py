import numpy as np


class KnnEuclidean:
    def __init__(self, k_neighbor=3):
        self.k_neighbor = k_neighbor

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        predictions = [self._predict(x) for x in np.array(X)]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X - x, axis=1) # Euclidean distance

        k_indices = np.argsort(distances)[:self.k_neighbor]

        k_nearest_labels = self.y[k_indices]

        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        most_common = unique[np.argmax(counts)]
        return most_common
