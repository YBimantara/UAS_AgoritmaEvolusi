import numpy as np
import pandas as pd
from firefly import firefly_algorithm
from kmeans import kmeans
from utils import calculate_sse

data = pd.read_csv("../data/dataset.csv").values

fireflies = [
    np.array([[1,2,1,2],[8,8,8,8],[4,5,4,5]]),
    np.array([[2,1,2,1],[9,9,9,8],[5,4,5,4]])
]

fireflies = firefly_algorithm(data, fireflies)
best_centroid = fireflies[0]

print("Centroid awal hasil Firefly:")
print(best_centroid)

final_centroid, labels = kmeans(data, best_centroid)

print("\nCentroid akhir:")
print(final_centroid)

print("\nCluster data:")
print(labels)

sse = calculate_sse(data, labels, final_centroid)
print("\nSSE:", sse)
