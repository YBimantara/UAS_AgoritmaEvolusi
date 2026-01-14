import numpy as np

def euclidean(a, b):
    return np.linalg.norm(a - b)

def calculate_sse(data, labels, centroids):
    sse = 0
    for i in range(len(data)):
        sse += euclidean(data[i], centroids[labels[i]]) ** 2
    return sse
