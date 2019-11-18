"""
"""

import numpy as np
import matplotlib.pyplot as plt


# Dataset
# --------------------------------------------------------
dataset1 = np.zeros((100,2))
dataset1[:, 0] = np.linspace(start=0, stop=99, num=100)
from sklearn.datasets.samples_generator import make_blobs
centers = [(100, 100)]
dataset2, y = make_blobs(n_samples=1000, cluster_std=10,
                  centers=centers, shuffle=False, random_state=42)
dataset2 = np.vstack((dataset1, dataset2))
data = dataset2
# --------------------------------------------------------



from dish_class import DiSH
algo = DiSH(mu=30, epsilon=1)
algo.data = data
algo.nr_of_features = data.shape[1]

point = data[120]
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], "bo")
ax.plot(point[0], point[1], "ro")

nr_neighbors_per_dim = algo._get_neighbor_count(point)
candidate_features = algo._get_candidate_attributes(nr_neighbors_per_dim)
best_subspace = algo._best_first_search(point, nr_neighbors_per_dim, candidate_features)

print(nr_neighbors_per_dim)
print(len(algo.__get_neighbors(point, features=[0, 1])))
print(candidate_features)
print(best_subspace)


preference_vector = algo._get_preference_vectors()

# PLOT PREFERNCE VECTORs
# -------------------------------------------------------------------
pref_ax0 = data[preference_vector[:, 0]]
pref_ax1 = data[preference_vector[:, 1]]

ax.plot(data[:, 0], data[:, 1], "ko", markersize=10)
ax.plot(pref_ax0[:, 0], pref_ax0[:, 1], "r^", label="w_up == True", markersize=10)
ax.plot(pref_ax1[:, 0], pref_ax1[:, 1], "y>", label="w_right == True", markersize=10)

