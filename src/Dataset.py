import pandas as pd
import os
import config
import numpy as np
from sklearn.cluster import KMeans as sk_KMeans
from tqdm import tqdm

def read_dataset(name, array=True):
    '''
    name of the file in the data folder without the extension
    '''
    # Load data
    X = pd.read_csv(os.path.join(config.DATA_FOLDER, f'{name}.csv'))
    X = X.sample(frac=1).reset_index(drop=True)
    y = X.pop('target')
    
    if name in ['breast', 'dermatology']:
        # Get dummies for categorical data
        # X = X.dropna(axis=0, how='any')
        # y = y[X.index]
        X = X.fillna(0)
        # X = pd.get_dummies(X)
        # X = X.fillna(X.median())
    elif name == 'wine':
        X.iloc[:, -1] /= 100
    

    if array:
        X = X.values
        y = y.values
    return X, y

def get_binary_dataset(X, k, r=100):
    '''
    Returns the binary dataset
    - X_b: Binary dataset with shape (n, total_k)
    - ls_partitions_labels: List of partitions labels
    - cluster_sizes: List of cluster sizes
    '''
    n = len(X)
    classes = k

    max_k = np.sqrt(n) + 1
    ls_partitions = []
    ls_partitions_labels = []
    cluster_sizes = []
    total_k = 0

    for _ in tqdm(range(r), desc="Clustering Progress"):
        k = np.random.randint(classes, max_k) # Closed form both sides
        partition_i = sk_KMeans(n_clusters=k, n_init=10).fit(X).labels_
        ls_partitions_labels.append(partition_i)
        cluster_sizes.append(k)
        ls_partitions.append(one_hot_encode(partition_i, k))
        total_k += k

    X_b = np.hstack(ls_partitions)
    # self.ls_partitions_labels = ls_partitions_labels
    # self.cluster_sizes = cluster_sizes

    print("X_b shape:", X_b.shape)
    assert X_b.shape == (n, total_k), "Error in shape of X_b"

    return X_b, ls_partitions_labels, cluster_sizes

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot