import numpy as np
from utils import euclidean, calculate_sse

def assign_cluster(data, centroids):
    labels = []
    for x in data:
        distances = [euclidean(x, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

def firefly_algorithm(data, fireflies, alpha=0.2, beta=0.5, max_iter=10):
    n_fireflies = len(fireflies)

    for _ in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if i != j:
                    labels_i = assign_cluster(data, fireflies[i])
                    labels_j = assign_cluster(data, fireflies[j])

                    sse_i = calculate_sse(data, labels_i, fireflies[i])
                    sse_j = calculate_sse(data, labels_j, fireflies[j])

                    if sse_j < sse_i:
                        fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i])                                        + alpha * np.random.randn(*fireflies[i].shape)
    return fireflies
