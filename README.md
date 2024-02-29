# KCC_K-Means_Consensus_Clustering

## Introduction

In this project, we present the implementation and evaluation of the K-means-based Consensus Clustering 
algorithm proposed by Wu et al. [1]. The algorithm is a consensus clustering method that combines multiple K-Means clusterings to obtain
a single, more stable one. The paper claims thatthe algorithm outperforms other consensus clustering
methods and is robust to noise and outliers.

**Objective**: Given a finite set of basic partitionings of the same dataset, obtain a single one which
agrees with them as much as possible. The paper employs mathematical demonstrations to derive 
utility functions, enabling the transformation of the consensus clustering problem into a K-Means
problem. Subsequently, the 2-phase algorithm employed in K-Means is utilized for solving the transformed problem

## Requirements

The project is implemented in Python 3.8.5.
To restore the environment, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

- Set the parameters in the `config.py` file (ids of the datasets in UCI repository, number of basic partitionings an i/o folders)
- Execute `run.sh`
  1. Download the datasets from the UCI repository (You can comment this line if you have already downloaded the datasets)
  2. Run the main file `main.py` with the parameters set in the `config.py` file
  3. The results will be saved in the results folder

## Dataset details

- [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+diagnostic)

- [Ecoli](https://archive.ics.uci.edu/dataset/39/ecoli)

- [iris](https://archive.ics.uci.edu/dataset/53/iris)

- [pendigits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)

- [satimage](https://datahub.io/machine-learning/satimage)

- [dermatology](https://archive.ics.uci.edu/dataset/33/dermatology)

- [wine](https://archive.ics.uci.edu/dataset/109/wine)

