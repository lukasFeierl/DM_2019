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
        print("Starting Dish.")
        print("epsiolon: \t", self.epsilon)
        print("mu: \t\t", self.mu)

        preference_vector = self._get_preference_vectors()       # basically w(p) for each row p in data
        pq = self.init_pq()
        for index in range(pq.shape[0]):
            p_index = int(pq[index, 0])                     # d and w are ordered according to data indices, ...
                                                            # but pq is ordered by RDIst!
            d1, d2 = self.get_subspace_distance(p_index, preference_vector)
            # TODO: Get Mu-Nearest Neighbor
            # TODO: Calc ReachDist
            pq = self.update_pq(pq, d1, d2)
            pq = self.sort_pq(pq, index)
        return pq                                           # equal to cluster order co.
    # -------------------------------------------------------------------

    #

    def _get_preference_vectors(self):
        """ calculates the preference vector of each point, describing which subspace-dimensions are relevant for it.
        The number of zero-values in the preference vector is called the subspace dimensionality.
        """
        preference_matrix = np.zeros(self.data.shape, dtype=bool)       # initialises. Later stores w(p) for each point
        for p_index, point in enumerate(self.data):
            is_best_subspace = self._get_best_subspace(point)
            preference_matrix[p_index][is_best_subspace] = True
        return preference_matrix

    def _get_best_subspace(self, point):
        """ calculates the best subspace for a given point. A attributes contribute to the subspace if enough
        elements are in the neighborhood of the point in a radius epsilon. """
        nr_neighbors_per_dim = self._get_neighbor_count(point)
        candidate_features = self._get_candidate_attributes(nr_neighbors_per_dim)
        best_subspace = self._best_first_search(point, nr_neighbors_per_dim, candidate_features)
        return best_subspace

    def _get_neighbor_count(self, point):
        """ counts how many datapoints are in the neighborhood of the point if projected along each feature/column of
        the data. Returns a vector for each row of the data"""
        neighbor_count = np.zeros(self.nr_of_features)                          # initialise vector
        for feature in range(self.nr_of_features):                              # per feature:
            neighbors = self._get_neighbors(point, features=[feature])              # get all neighbors
            neighbor_count[feature] = len(neighbors)                                # count amount
        return neighbor_count

    def _get_neighbors(self, point, features):
        """ returns all neighbors of a point, if projected along features in a radius epsilon"""
        is_near = self.DIST(point[features], self.data[:, features]) <= self.epsilon      # data[features] is a projection
        return self.data[is_near]

    def DIST(self, point, data):
        """ calculates the euclidean distance between point and all points in data"""
        return np.sqrt(np.sum((point - data)**2, axis=1))       # eucleadian norm

    def DIST_projected(self, point, data, feature_matrix):
        dist_matrix = (point - data) ** 2
        projected_dist_matrix = np.multiply(dist_matrix, feature_matrix)
        final_dist_vector = np.sqrt(np.sum(projected_dist_matrix, axis=1))
        return final_dist_vector

    def _best_first_search(self, point, nr_neighbors_per_dim, candidate_features):
        nr_neighbors_per_dim = nr_neighbors_per_dim.copy()
        best_subspace = []
        for index in range(len(candidate_features)):
            best_feature = nr_neighbors_per_dim.argmax()
            nr_neighbors_per_dim[best_feature] = -1                 # set "visited" features to "already searched"
            proposed_combination = best_subspace + [best_feature]
            if len(self._get_neighbors(point, features=proposed_combination)) >= self.mu:
                best_subspace = proposed_combination
        return best_subspace

    def _get_candidate_attributes(self, nr_of_neighbors_per_dim):
        is_candidate = (nr_of_neighbors_per_dim >= self.mu)
        candidate_features = [i for i, val in enumerate(is_candidate) if True]
        return candidate_features

    #

    def init_pq(self):
        pq = np.zeros((self.data.shape[0], 3))
        reach_dist_d1 = np.full(self.data.shape[0], fill_value=np.NaN)
        reach_dist_d2 = reach_dist_d1.copy()
        index = np.arange(0, data.shape[0])

        pq[:, 0] = index
        pq[:, 1] = reach_dist_d1
        pq[:, 2] = reach_dist_d2
        return pq

    #

    def update_pq(self, pq, d1, d2):
        pq_sorted = pq[np.argsort(pq[:, 0])]                # sort in order of elements in data (data index)

        d1_old_new = np.vstack((pq_sorted[:, 1], d1))       # Append previous and new d to matrix, where
        d2_old_new = np.vstack((pq_sorted[:, 2], d2))       # [0,:] <- old values     [1,:] <- new values

        # if d1 are equal, d2 is minimized
        d1_same = (d1_old_new[0] == d1_old_new[1])
        pq_sorted[:, 2][d1_same] = np.nanmin(d2_old_new, axis=0)[d1_same]

        # if d1 are unequal, d1 and d2 are taken from the one where d1 is minimum.
        minimum_value = np.nanmin(d1_old_new, axis=0)
        is_minimum = np.zeros(d1_old_new.shape, dtype=bool)
        is_minimum[d1_old_new == minimum_value] = True
        is_minimum[:, d1_same] = False
        pq_sorted[:, 1][~d1_same] = d1_old_new[is_minimum]
        pq_sorted[:, 2][~d1_same] = d2_old_new[is_minimum]

        # resort such that order follows the order of the pq again (pq index)
        return pq_sorted[np.array(pq[:, 0], dtype="int64")]

    def sort_pq(self, pq, index):
        pq_sort = pq[index + 1:]                                    # only sort not yet visited data-points
        pq_argsort = np.lexsort((pq_sort[:, 2], pq_sort[:, 1]))     # Sort according to d1, then d2
        pq[index + 1:] = pq_sort[pq_argsort]                        # Apply reordering
        return pq

    #

    def get_subspace_distance(self, p_index, preference_vector):
        w_p = preference_vector[p_index]
        W_q = preference_vector
        W_pq = W_q * w_p
        Lambda_pq = W_pq.shape[1] - W_pq.sum(axis=1)

        is_included_p = (w_p == W_pq).all(axis=1)
        is_included_q = (preference_vector == W_pq).all(axis=1)
        is_included = is_included_q + is_included_p

        feature_matrix = W_pq
        is_parallel = self.DIST_projected(data[p_index], data, feature_matrix) > 2 * self.epsilon
        delta_pq = is_included * is_parallel

        W_inverse = ~W_pq
        d2 = self.DIST_projected(data[p_index], data, W_inverse)
        d1 = Lambda_pq + delta_pq
        return (d1, d2)  # np.vstack()

    #

    def update_pq(self, pq, d1, d2):
        # Append to Matrix      [0,:] <- old values     [1,:] <- new values
        d1_old_new = np.vstack((pq[:, 1], d1))
        d2_old_new = np.vstack((pq[:, 2], d2))

        # if d1 are equal, d2 is minimized
        d1_same = (d1_old_new[0] == d1_old_new[1])
        pq[:, 2][d1_same] = np.nanmin(d2_old_new, axis=0)[d1_same]

        # if d1 are unequal, d1 and d2 are taken from the one where d1 is minimum.
        minimum_value = np.nanmin(d1_old_new, axis=0)
        is_minimum = np.zeros(d1_old_new.shape, dtype=bool)
        is_minimum[d1_old_new == minimum_value] = True
        is_minimum[:, d1_same] = False

        pq[:, 1][~d1_same] = d1_old_new[is_minimum]
        pq[:, 2][~d1_same] = d2_old_new[is_minimum]
        return pq

    #

    def SDIST(self, data, preference_vector, row_p, row_q, epsilon=0.1):
        w_p = preference_vector[row_p]
        w_q = preference_vector[row_q]

        w_pq = w_p and w_q
        lambda_pq = w_pq.sum()

        is_included_p = (w_pq == w_p).all()
        is_included_q = (w_pq == w_q).all()

        features = [i for i, val in enumerate(w_pq) if True]
        is_parallel = (self.DIST(data[row_p][features], data[row_q][features][np.newaxis]) > 2 * epsilon).all()

        delta_pq = (is_included_p or is_included_q) and is_parallel

        features_inverse = [i for i, val in enumerate(w_pq) if False]
        d2 = self.DIST(data[row_p][features_inverse], data[row_q][features_inverse][np.newaxis])[0]
        d1 = lambda_pq + delta_pq
        return (d1, d2)

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

