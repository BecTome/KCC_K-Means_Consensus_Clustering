from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import config

output_folder = config.DATA_FOLDER
dataset_ids = config.DATASET_IDS

# Function to export dataset to CSV
def export_to_csv(data, target, metadata, output_folder):
    '''
    Receives a dataset, its target, and its metadata, and exports it to a CSV file.
    Adapted for UCI datasets but extensible to other sources.
    '''
    # Create a DataFrame for features and targets
    data['target'] = target
    name_to_write = metadata['name'].split(' ')[0].lower()

    # Export to CSV
    csv_file = f"{output_folder}/{name_to_write}.csv"
    data.to_csv(csv_file, index=False)

    print(f"Dataset '{name_to_write}' exported to '{csv_file}'.")


if __name__ == "__main__":
    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fetch datasets, export to CSV, and print metadata
    ls_info = []
    for dataset_id in dataset_ids:
        dataset = fetch_ucirepo(id=dataset_id)
        export_to_csv(dataset.data.features, dataset.data.targets, dataset.metadata, output_folder)

        # Export to a file called info.csv the number of Objects, Attributes and Classes for each
        info = pd.DataFrame({   'Name': [dataset.metadata['name'].split(' ')[0].lower()],
                                'Objects': [dataset.data.features.shape[0]],
                                'Attributes': [dataset.data.features.shape[1] - 1],
                                'Classes': [len(np.unique(dataset.data.targets))]})

        ls_info.append(info)
    
    info = pd.concat(ls_info)
    info.to_csv(f"{output_folder}/info.csv", index=False)