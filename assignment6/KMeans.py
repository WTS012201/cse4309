# William Sigala
# ID: 1001730022

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

colors = ListedColormap(["red", "green", "blue", "orange"])

DATASET_FILE = "ClusteringData.txt"
K = 2


def rmse(x1, x2):
    return np.square(np.sum((x1 - x2)**2))


def estimate_centroid(samples, centroids):
    candidate_solutions, errors = [], []

    for sample_idx in range(samples.shape[0]):
        centroid_errors = np.array([])
        for centroid in centroids:
            error = rmse(centroid, samples.iloc[sample_idx, :])
            centroid_errors = np.append(centroid_errors, error)

        min_error = np.amin(centroid_errors)
        candidate = np.where(centroid_errors == min_error)[0][0]

        candidate_solutions.append(candidate)
        errors.append(min_error)

    return candidate_solutions, errors


def plot(results, centroids, round):
    title = f"{len(results)} samples for {len(centroids)} class Ks, round {round}"
    _, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(results.iloc[:, 0], results.iloc[:, 1],  marker='o',
                c=results.iloc[:, 2],
                cmap=colors, s=10, alpha=0.2)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='s', s=100, c=list(range(centroids.shape[0])),
                cmap=colors)
    ax.set_xlabel(r'x', fontsize=14)
    ax.set_ylabel(r'y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title)
    plt.savefig(f"k{len(centroids)}_round_{round}")


def KMeans(datasetFile, K):
    df = pd.read_csv(datasetFile, header=None)
    samples = df.iloc[:, :-1]

    centroids = samples.sample(K).to_numpy()
    it = 0
    max_it = 9
    result = None

    while it < max_it:
        sample_scores = estimate_centroid(samples, centroids)
        result = pd.concat([samples, pd.DataFrame(
            np.array(sample_scores).T, columns=["centroid", "error"])], axis=1)
        centroids = result.groupby("centroid").agg(
            'mean').iloc[:, :2].reset_index(drop=True).to_numpy()

        if max_it // 2 == it or it == 0:
            plot(result, centroids, it + 1)
        it += 1

    plot(result, centroids, it)


if __name__ == "__main__":
    datasetFile = sys.argv[1] if len(sys.argv) > 1 else DATASET_FILE
    k = int(sys.argv[2]) if len(sys.argv) == 3 else K

    KMeans(datasetFile, k)
    KMeans(datasetFile, k + 1)
    KMeans(datasetFile, k + 2)
