
from src.Dataset import read_dataset, get_binary_dataset
from sklearn.metrics.cluster import adjusted_rand_score
from src.ConsensusKMeans import ConsensusKMeans
import numpy as np
import config
import datetime
import pandas as pd
import os
import time

ls_datasets = ['breast', 'ecoli', 'iris', 'pen-based', 'statlog', 'dermatology', 'wine'] # 'pendigits', takes too long
d_results = {}
d_time = {}
for dataset in ls_datasets:
    # Load data
    X, y = read_dataset(dataset)
    k = len(np.unique(y))
        
    # Get the binary dataset
    # tic = time.time()
    binary_info = get_binary_dataset(X, k, r=100)
    toc = time.time()
    # d_binary_generation_time[dataset] = toc - tic
    
    # Instantiate ConsensusKMeans for every distance
    KCC = ConsensusKMeans(n_clusters=k, type='Uc', normalize=False)

    # Hierarchical sklearn clustering
    from sklearn.cluster import AgglomerativeClustering
    HC = AgglomerativeClustering(n_clusters=k, linkage='ward')

    # Train the models and measure the time
    tic = time.time()
    KCC.fit(*binary_info)
    toc = time.time()
    d_time[dataset] = {}
    d_time[dataset]['KCC'] = toc - tic
    print(f"Time for {dataset} KCC: {toc - tic}")

    tic = time.time()
    HC.fit(X)
    toc = time.time()
    d_time[dataset]['HC'] = toc - tic
    print(f"Time for {dataset} HC: {toc - tic}")


    # Measure the performance
    d_results[dataset] = {}
    d_results[dataset]['KCC'] = adjusted_rand_score(y, KCC.labels_)
    d_results[dataset]['HC'] = adjusted_rand_score(y, HC.labels_)

    print(f"Adjusted Rand Index for {dataset} KCC: {adjusted_rand_score(y, KCC.labels_)}")
    print(f"Adjusted Rand Index for {dataset} HC: {adjusted_rand_score(y, HC.labels_)}")
    
# Print the results
df_results = pd.DataFrame(d_results).T
# df_results = df_results.loc[:, ['Uc', 'Uh', 'Ucos', 'ULp5', 'ULp8', 'NUc', 'NUh', 'NUcos', 'NULp5', 'NULp8']]
print("RESULTS\n",df_results, "\n")

df_time = pd.DataFrame(d_time).T
# df_time = df_time.loc[:, ['Uc', 'Uh', 'Ucos', 'ULp5', 'ULp8', 'NUc', 'NUh', 'NUcos', 'NULp5', 'NULp8']]
# df_time['KMeans_100times'] = d_binary_generation_time.values()
print("TIMES\n",df_time, "\n")

# Save the results to results path and create a csv with the results and timestamp
now = datetime.datetime.now()

os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
df_results.to_csv(f"{config.RESULTS_FOLDER}/COMPARISON_results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
df_time.to_csv(f"{config.RESULTS_FOLDER}/COMPARISON_time_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")

print(f"Results saved to {config.RESULTS_FOLDER}/COMPARISON_results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
print(f"Times saved to {config.RESULTS_FOLDER}/COMPARISON_time_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
