############################################
# This script runs the ConsensusKMeans algorithm for the datasets in the list ls_datasets
# It saves the results to the results folder and creates a csv with the results and timestamp
# It also saves the times to the results folder and creates a csv with the times and timestamp
# The results are printed to the console

# Author: Alberto Becerra Tomé
# Date: 2024/02/29
# GitHub User: BecTome

############################################

if __name__ == "__main__":

    # Imports
    from sklearn.metrics.cluster import adjusted_rand_score
    import numpy as np
    import config
    import datetime
    import pandas as pd
    import os
    import time

    # Local imports
    from src.Dataset import read_dataset, get_binary_dataset
    from src.ConsensusKMeans import ConsensusKMeans



    ls_datasets = ['breast', 'ecoli', 'iris', 'pen-based', 'statlog', 'dermatology', 'wine']
    # ls_datasets = ['breast', 'dermatology']
    d_results = {}
    d_time = {}
    d_binary_generation_time = {}
    np.random.seed(100)
    for dataset in ls_datasets:
        # Load data
        X, y = read_dataset(dataset)
        k = len(np.unique(y)) # Set K as the number of classes

        # Get the binary dataset
        tic = time.time()
        # binary_info = get_binary_dataset(X, k, r=100)
        X_b, _, cluster_sizes = get_binary_dataset(X, k, r=100)
        toc = time.time()

        d_binary_generation_time[dataset] = toc - tic
        
        # Instantiate ConsensusKMeans for every distance
        KCC_uc = ConsensusKMeans(n_clusters=k, type='Uc', normalize=False, cluster_sizes=cluster_sizes)
        KCC_ucos = ConsensusKMeans(n_clusters=k, type='Ucos', normalize=False, cluster_sizes=cluster_sizes)
        KCC_uh = ConsensusKMeans(n_clusters=k, type='Uh', normalize=False, cluster_sizes=cluster_sizes)
        KCC_ulp5 = ConsensusKMeans(n_clusters=k, type='ULp', p=5, normalize=False, cluster_sizes=cluster_sizes)
        KCC_ulp8 = ConsensusKMeans(n_clusters=k, type='ULp', p=8, normalize=False, cluster_sizes=cluster_sizes)

        # Instantiate ConsensusKMeans for normalized utility functions
        KCC_uc_norm = ConsensusKMeans(n_clusters=k, type='Uc', normalize=True, cluster_sizes=cluster_sizes)
        KCC_ucos_norm = ConsensusKMeans(n_clusters=k, type='Ucos', normalize=True, cluster_sizes=cluster_sizes)
        KCC_uh_norm = ConsensusKMeans(n_clusters=k, type='Uh', normalize=True, cluster_sizes=cluster_sizes)
        KCC_ulp5_norm = ConsensusKMeans(n_clusters=k, type='ULp', p=5, normalize=True, cluster_sizes=cluster_sizes)
        KCC_ulp8_norm = ConsensusKMeans(n_clusters=k, type='ULp', p=8, normalize=True, cluster_sizes=cluster_sizes)

        # create a list with all the models
        models = [KCC_uc, KCC_uh, KCC_ucos,  KCC_ulp5, KCC_ulp8, 
                  KCC_uc_norm, KCC_uh_norm, KCC_ucos_norm,  KCC_ulp5_norm, KCC_ulp8_norm]
        
        # Fit all the models and get the ARI. Create a dataframe with the results having 
        # the dataset name as index and the distance as columns. Print nice the results

        d_results[dataset] = {} # Store the results for each model and dataset
        d_time[dataset] = {}

        for model in models:

            # Fit the model and get the ARI
            tic = time.time()
            model.fit(X_b)        
            toc = time.time()     

            ari = adjusted_rand_score(y, model.labels_)

            model_type = model.type if model.normalize == False else 'N' + model.type

            if 'Lp' in model_type:
                # Add the 5 and the 8 to the Lp models
                model_type += str(model.p)

            d_results[dataset][model_type] = ari
            d_time[dataset][model_type] = toc - tic

            print(f"ARI for {dataset} and {model_type} is {ari} ({toc - tic:.3f} seconds)")

        print("\n")


    # Print the results
    df_results = pd.DataFrame(d_results).T
    df_results = df_results.loc[:, ['Uc', 'Uh', 'Ucos', 'ULp5', 'ULp8', 'NUc', 'NUh', 'NUcos', 'NULp5', 'NULp8']]
    print("RESULTS\n",df_results, "\n")

    df_time = pd.DataFrame(d_time).T
    df_time = df_time.loc[:, ['Uc', 'Uh', 'Ucos', 'ULp5', 'ULp8', 'NUc', 'NUh', 'NUcos', 'NULp5', 'NULp8']]
    df_time['KMeans_100times'] = d_binary_generation_time.values()
    print("TIMES\n",df_time, "\n")

    # Save the results to results path and create a csv with the results and timestamp
    now = datetime.datetime.now()

    os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
    df_results.to_csv(f"{config.RESULTS_FOLDER}/results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    df_time.to_csv(f"{config.RESULTS_FOLDER}/time_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")

    print(f"Results saved to {config.RESULTS_FOLDER}/results_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    print(f"Times saved to {config.RESULTS_FOLDER}/time_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
