import numpy as np
from utils import euclidean

def kmeans(data, centroids, max_iter=100):
    for iteration in range(max_iter):
        labels = []
        for x in data:
            distances = [euclidean(x, c) for c in centroids]
            labels.append(np.argmin(distances))
        labels = np.array(labels)

        new_centroids = []
        for k in range(len(centroids)):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[k])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            print(f"Convergent at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return centroids, labels
