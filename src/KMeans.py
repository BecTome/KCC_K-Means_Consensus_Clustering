import numpy as np
import numpy_indexed as npi
from scipy.spatial.distance import cdist

class KMeans:

    def __init__(self, n_clusters, distance='euclidean', n_init=10, max_iter=1000, tol=1e-4, random_state=None):
        self.k = n_clusters
        self.distance = distance # 'euclidean', 'jensenshannon', 'cosine', 'p'
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def initialize_centroids(self, X):
        idxs = np.random.choice(X.shape[0], self.k)
        return X[idxs, :]
    
    @staticmethod
    def update_centroids(X, centroids_old, labels):
        centroids = centroids_old.copy()
        for label in np.unique(labels):
            centroids[label] = np.mean(X[labels==label])
        return centroids

    def fit(self, X):
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            if self.random_state is not None:
                np.random.seed(self.random_state)

            centroids = self.initialize_centroids(X)
            it = 0
            while True:
                d_matrix = cdist(X, centroids, metric=self.distance)
                labels = d_matrix.argmin(axis=1)
                centroids_old = centroids
                centroids = self.update_centroids(X, centroids_old, labels)
                it += 1

                if ((np.linalg.norm(centroids - centroids_old, 1, axis=1).min() < self.tol) | (it > self.max_iter)):
                    break

            # Calculate inertia for the current run
            current_inertia = np.sum(d_matrix.min(axis=1))

            # Update best result if current inertia is lower
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels_ = best_labels
        return self