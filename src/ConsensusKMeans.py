import numpy as np
import numpy_indexed as npi
from scipy.spatial.distance import cdist
import config
from scipy.stats.contingency import crosstab
from tqdm import tqdm
from scipy.stats import entropy

class ConsensusKMeans:

    r = config.R

    def __init__(self, n_clusters, cluster_sizes, n_init=10, max_iter=1000, tol=1e-10, 
                 random_state=None, type='Uc', p=5, normalize=False):
        self.k = n_clusters
        # self.distance = distance # 'euclidean', 'jensenshannon', 'cosine', 'p'
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.cluster_sizes = cluster_sizes
        self.ls_partitions_labels = []
        self.type = type
        self.normalize = normalize
        self.p = p
        # self.weights = weights if weights is not None else np.ones(self.r) / self.r

        if type == 'Uc':
            self.d_dist = {'metric': 'euclidean'}
        elif type == 'Ucos':
            self.d_dist = {'metric': 'cosine'}
        elif type == 'ULp':
            self.d_dist = {'metric': 'minkowski', 'p': p}

    def initialize_centroids(self, X):
        X_unique = np.unique(X, axis=0)
        idxs = np.random.choice(X_unique.shape[0], self.k)
        return X_unique[idxs, :]
    
    @staticmethod
    def extract_p(pi, pi_i):
        '''
        The information that we need to extract from the contingency table is:
            - $p_{kj}^{(i)}$ = the ratio of objects in consensus cluster k that belong to cluster j based on partition i
            - $p_k$ = the number of objects in consensus cluster k (p_k+)
            - $P_k^{i}$ = $p_{kj}^{(i)}/p_k$
            - $P^(i) = the vector with the ratios of objects in each cluster based on partition i
        '''
        # Calculate contingency matrix (n)
        cont_i = crosstab(pi, pi_i).count

        # Get p-contingency matrix
        p_cont_i = cont_i / cont_i.sum()

        # Calculate the p_k for each cluster in PI
        p_k = p_cont_i.sum(axis=1)
        p_k = p_k.reshape(-1, 1)

        # Get the P_k^(i)
        P_k_i = p_cont_i / p_k

        # This is a constant and it's the ratio of points in each cluster in pi_i
        P_i = p_cont_i.sum(axis=0)

        return p_k, P_k_i, P_i
    
    def update_centroids(self, centroids_old, labels):
        centroids = centroids_old.copy()
        for j, k in enumerate(np.unique(labels)):
            ls_i = []
            for pi_i in self.ls_partitions_labels:
                _, P_k_i, _ = self.extract_p(labels, pi_i)
                # print(P_k_i.shape)
                ls_i.extend(P_k_i[j, :])
            centroids[k, :] = ls_i
            
        return centroids
    
    @staticmethod
    def compute_kl_divergence_matrix(data, centroids, eps=1E-10):
        """
        Compute KL divergence matrix using broadcasting.

        Parameters:
        - data: Input data array with shape (m, n)
        - centroids: Centroids array with shape (k, n)
        - eps: Small value added to data and centroids to avoid log(0)

        Returns:
        - kl_divergence_matrix: Resulting KL divergence matrix with shape (m, k)
        """
        expanded_data = np.expand_dims(data + eps, axis=1)
        expanded_centroids = np.expand_dims(centroids + eps, axis=0)

        kl_divergence_matrix = np.sum(expanded_data * (np.log2(expanded_data / expanded_centroids)), axis=2)

        return kl_divergence_matrix
    
    def f(self, X_b, centroids, cluster_sizes, weights):

        X_b_split = np.split(X_b, np.cumsum(cluster_sizes), axis=1)[:-1]
        centroids_split = np.split(centroids, np.cumsum(cluster_sizes), axis=1)[:-1]
        eps = 1E-8
        pw = 1
        if self.type == 'Uc':
            pw = 2

        if self.type != 'Uh':
            if self.normalize:
                weight_norm = 1/(self.get_weight_norm(X_b_split, centroids_split) + eps)
                X_r = [cdist(X_b_split[i], centroids_split[i], **self.d_dist) ** pw * weight_norm[i] for i in range(len(X_b_split))]
            else:
                X_r = [cdist(X_b_split[i], centroids_split[i], **self.d_dist) for i in range(len(X_b_split))]
        else:
            X_r = []
            if self.normalize:
                weight_norm = 1/(self.get_weight_norm(X_b_split, centroids_split) + eps)
                X_r = [self.compute_kl_divergence_matrix(X_b_split[i], centroids_split[i], eps=eps) * weight_norm[i] for i in range(len(X_b_split))]
            else:
                X_r = [self.compute_kl_divergence_matrix(X_b_split[i], centroids_split[i], eps=eps) for i in range(len(X_b_split))]

        return np.average(X_r, weights=weights, axis=0)

    def fit(self, X_b, weights=None):

        cluster_sizes = self.cluster_sizes
        X_b_split = np.split(X_b, np.cumsum(cluster_sizes), axis=1)[:-1]
        self.ls_partitions_labels = [np.argmax(x, axis=1) for x in X_b_split]

        if weights is None:
            weights = np.ones(self.r) / self.r
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for _ in tqdm(range(self.n_init)):
            if self.random_state is not None:
                np.random.seed(self.random_state)

            centroids = self.initialize_centroids(X_b)
            # labels = np.random.choice(self.k, len(X_b))
            it = 0
            inertia_old = 1E8
            while True:
                d_matrix = self.f(X_b, centroids, self.cluster_sizes, weights=weights)
                labels = d_matrix.argmin(axis=1)
                centroids_old = centroids
                centroids = self.update_centroids(centroids_old, labels)
                it += 1

                inertia = d_matrix.min(axis=1).sum()
                if (inertia_old - inertia) < self.tol:
                    # print(f"Converged in {it} iterations")
                    break

                inertia_old = inertia

            # Calculate consensus for the current run
            current_inertia = d_matrix.min(axis=1).sum()

            # Update best result if current inertia is lower
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels_ = best_labels
        # self.utility = self.get_utility(self.labels_, self.ls_partitions_labels)
        self.weights = weights
        return self
    
    def predict(self, X_b):
        return self.f(X_b, self.centroids, self.cluster_sizes, weights=self.weights).argmin(axis=1)
    
    def fit_predict(self, X_b, **kwargs):
        self.fit(X_b, **kwargs)
        return self.labels_

    
    def get_weight_norm(self, X_b_split, centroids_split):
        weight_norm = np.ones((self.r, self.k))
        for i in range(len(X_b_split)):
            P_i = np.unique(self.ls_partitions_labels[i], return_counts=True)[1] # / len(self.ls_partitions_labels[i])
            
            if self.type in 'Uc':            
                weight_norm[i, :] = [np.linalg.norm(centroid, ord=2) ** 2 for centroid in centroids_split[i]]
                weight_norm[i, :] -= np.linalg.norm(P_i, ord=2) ** 2
                weight_norm[i, :] = np.abs(weight_norm[i, :])
            elif self.type == 'Ucos':
                weight_norm[i, :] = [np.linalg.norm(centroid, ord=2) for centroid in centroids_split[i]]
                weight_norm[i, :] -= np.linalg.norm(P_i, ord=2)
                weight_norm[i, :] = np.abs(weight_norm[i, :])
            elif self.type == 'ULp':
                weight_norm[i, :] = [np.linalg.norm(centroid, ord=self.p) for centroid in centroids_split[i]]
                weight_norm[i, :] -= np.linalg.norm(P_i, ord=self.p)
                weight_norm[i, :] = np.abs(weight_norm[i, :])
            elif self.type == 'Uh':
                eps = 1E-10
                entropy_mk_i = np.array([entropy(centroid + eps, base=2) for centroid in centroids_split[i]])
                entropy_P_i = entropy(P_i + eps, base=2)
                weight_norm[i, :] = (- entropy_mk_i) - (- entropy_P_i)
                
        return weight_norm