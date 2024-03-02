
from src.Dataset import read_dataset, get_binary_dataset
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from src.ConsensusKMeans import ConsensusKMeans
import numpy as np
import config
import datetime
import pandas as pd
import os
import time

ls_datasets = ['breast', 'ecoli', 'iris', 'pen-based', 'statlog', 'dermatology', 'wine']
d_results = {}
d_time = {}
ls_results = []
ls_time = []
for dataset in ls_datasets:
    d_results[dataset] = {}
    d_time[dataset] = {}

    # Load data
    X, y = read_dataset(dataset)
    k = len(np.unique(y))
        
    # Get the binary dataset
    # tic = time.time()
    X_b, _, cluster_sizes = get_binary_dataset(X, k, r=100)
    toc = time.time()
    # d_binary_generation_time[dataset] = toc - tic
    
    # Instantiate ConsensusKMeans for every distance
    KCC = ConsensusKMeans(n_clusters=k, type='Uh', normalize=False, cluster_sizes=cluster_sizes, n_init=10)

    # Hierarchical sklearn clustering

    HC_ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    HC_avg = AgglomerativeClustering(n_clusters=k, linkage='average')
    HC_single = AgglomerativeClustering(n_clusters=k, linkage='single')
    HC_comp = AgglomerativeClustering(n_clusters=k, linkage='complete')

    # Prototype based clustering
    KM = KMeans(n_clusters=k, n_init=10)

    # Gaussian Mixture Model
    GMM_f = GaussianMixture(n_components=k, covariance_type='full', n_init=10)
    GMM_t = GaussianMixture(n_components=k, covariance_type='tied', n_init=10)
    GMM_d = GaussianMixture(n_components=k, covariance_type='diag', n_init=10)
    GMM_s = GaussianMixture(n_components=k, covariance_type='spherical', n_init=10)

    # Spectral Clustering
    # SC = SpectralClustering(n_clusters=k, n_init=10)

    d_models = {'HC_ward': HC_ward, 'HC_avg': HC_avg, 'HC_single': HC_single, 'HC_comp': HC_comp, 
                'KM': KM, 'GMM_full': GMM_f, 'GMM_tied': GMM_t, 'GMM_diag': GMM_d, 'GMM_spher': GMM_s, # 'SC': SC,
                'KCC': KCC}

    for model_name, model in d_models.items():
        tic = time.time()
        labels = model.fit_predict(X if model_name != 'KCC' else X_b)
        toc = time.time()
        d_time[dataset][model_name] = toc - tic
        print(f"Time for {dataset} {model_name}: {toc - tic}")

        d_results[dataset][model_name] = adjusted_rand_score(y,labels)
        print(f"Adjusted Rand Index for {dataset} {model_name}: {adjusted_rand_score(y, labels)}")

    df_results = pd.DataFrame(d_results).T
    df_time = pd.DataFrame(d_time).T

    print(f"Results for {dataset}:\n{df_results}\n")
    print(f"Times for {dataset}:\n{df_results}\n")

# Print the results
# df_results = pd.concat(ls_results)

# df_results = df_results.loc[:, ['Uc', 'Uh', 'Ucos', 'ULp5', 'ULp8', 'NUc', 'NUh', 'NUcos', 'NULp5', 'NULp8']]
print("RESULTS\n",df_results, "\n")

# df_time = pd.concat(ls_time)
# df_time = df_time.loc[:, ['Uc', 'Uh', 'Ucos', 'ULp5', 'ULp8', 'NUc', 'NUh', 'NUcos', 'NULp5', 'NULp8']]
print("TIMES\n",df_time, "\n")

# Save the results to results path and create a csv with the results and timestamp
now = datetime.datetime.now()

os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
df_results.to_csv(f"{config.RESULTS_FOLDER}/COMPARISON_results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
df_time.to_csv(f"{config.RESULTS_FOLDER}/COMPARISON_time_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")

print(f"Results saved to {config.RESULTS_FOLDER}/COMPARISON_results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
print(f"Times saved to {config.RESULTS_FOLDER}/COMPARISON_time_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
