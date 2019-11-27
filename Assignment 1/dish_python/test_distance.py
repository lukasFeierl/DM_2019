"""
"""

import numpy as np
import matplotlib.pyplot as plt


def test_distance(dataset, p_index, q_index, result):
    print("Test Distance")
    from dish_class import DiSH
    algo = DiSH(epsilon=0.1, mu=3)
    algo.data = dataset

    point = dataset[p_index]
    data = dataset[q_index:q_index+1]

    distance = algo._DIST(point, data)

    if result != distance[0]:
        plt.plot(dataset[:, 0], dataset[:, 1], 'ko', alpha=0.5)
        plt.plot(dataset[p_index, 0], dataset[p_index, 1], 'ro')
        plt.plot(dataset[q_index, 0], dataset[q_index, 1], 'rx')
        raise ValueError("WRONG CALCULATION")

    print("succesfull")
    return


# Dataset 1
# --------------------------------------------------------
dataset1 = np.zeros((100,2))
dataset1[:, 0] = np.linspace(start=0, stop=99, num=100)
# --------------------------------------------------------


p_index = 0
q_index = 99
dataset = dataset1
result = 99
test_distance(dataset, p_index, q_index, result)


p_index = 0
q_index = 50
dataset = dataset1
result = 50
test_distance(dataset, p_index, q_index, result)


from sklearn.datasets import make_circles

# Dataset 1
# --------------------------------------------------------
from sklearn.datasets.samples_generator import make_blobs
centers = [(100, 100)]
dataset2, y = make_blobs(n_samples=1000, cluster_std=10,
                  centers=centers, shuffle=False, random_state=42)
dataset2 = np.vstack((dataset1, dataset2))
# --------------------------------------------------------

p_index = 0
q_index = 120
dataset = dataset2
result = np.sqrt(dataset2[q_index][0]**2 + dataset2[q_index][1]**2)
test_distance(dataset, p_index, q_index, result)
