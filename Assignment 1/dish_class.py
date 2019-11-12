"""
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DiSH():
    def __init__(self, epsilon, mu):
        self.epsilon = epsilon
        self.mu = mu
        self.data = None
        self.nr_of_features = None
        return

    #

    # -------------------------------------------------------------------
    def fit(self, data):
        self.data = data
        self.nr_of_features = self.data.shape[1]

        # Calculation
        preference_matrix = self._get_preference_matrix()       # basically w(p) for each row p in data

        return preference_matrix
    # -------------------------------------------------------------------

    #

    def _get_preference_matrix(self):
        preference_matrix = np.zeros(self.data.shape, dtype=bool)          # stores w(p) for each p
        for row, point in enumerate(self.data):
            is_best_subspace = self._get_best_subspace(point)

            preference_matrix[row][is_best_subspace] = True

        return preference_matrix

    #

    def _get_neighbor_count(self, point):
        """ counts how many datapoints are in the neighborhood of 'point'
        """
        neighbor_count = np.zeros(self.nr_of_features)                          # initialise vector
        for feature in range(self.nr_of_features):                              # per feature:
            neighbors = self._get_neighbors(point, features=[feature])              # get all neighbors
            neighbor_count[feature] = len(neighbors)                                # count amount
        return neighbor_count

    #

    def _get_neighbors(self, point, features):
        is_near = self.DIST(point[features], self.data[:, features]) <= self.epsilon      # data[features] is a projection
        return self.data[is_near]

    #

    def DIST(self, point, data):
        return np.sqrt(np.sum((point - data) ** 2, axis=1))  # euclidean norm

    #

    def _get_best_subspace(self, point):
        nr_neighbors_per_dim = self._get_neighbor_count(point)
        is_candidate = (nr_neighbors_per_dim >= self.mu)

        best_subspace = []
        candidate_features = [i for i, val in enumerate(is_candidate) if True]

        for index in range(len(candidate_features)):
            # Select new (best) candidate
            best_feature = nr_neighbors_per_dim.argmax()
            nr_neighbors_per_dim[best_feature] = -1
            proposed_combination = best_subspace + [best_feature]

            if len(self._get_neighbors(point, features=proposed_combination)) >= self.mu:
                best_subspace = proposed_combination

        return best_subspace


#


if __name__ == '__main__':

    # Get Data
    # -------------------------------------------------------------------
    fpath = r"./mouse.csv"
    dataframe = pd.read_csv(fpath, sep=" ", comment="#", header=None)
    data = dataframe.values
    # -------------------------------------------------------------------

    # FIT
    # -------------------------------------------------------------------
    algo = DiSH(epsilon=0.1, mu=40)

    w = algo.fit(data)

    plt.plot(data[:, 0], data[:, 1], 'bo')
    plt.plot(data[w[:,1], 0], data[w[:,1], 1], 'go')
    plt.plot(data[w[:,0], 0], data[w[:,0], 1], 'ro')
    plt.plot(data[w.all(axis=1), 0], data[w.all(axis=1), 1], 'ko')

    # -------------------------------------------------------------------

