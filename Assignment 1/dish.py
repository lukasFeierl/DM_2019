"""
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PARAMETER
# ----------------------
fpath = r"./mouse.csv"
# ----------------------


def dish(data, mu=4, epsilon=0.1):

    from dish_class import DiSH
    algo = DiSH(epsilon=epsilon, mu=mu)
    algo.data = data

    preference_vector = algo.fit(data)


    # PLOT PREFERNCE VECTORs
    # -------------------------------------------------------------------
    pref_ax0 = data[preference_vector[:, 0]]
    pref_ax1 = data[preference_vector[:, 1]]

    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], "ko", markersize=10)
    ax.plot(pref_ax0[:, 0], pref_ax0[:, 1], "r^", label="w_up == True", markersize=10)
    ax.plot(pref_ax1[:, 0], pref_ax1[:, 1], "y>", label="w_right == True", markersize=10)

    # Plot neighbors of a point
    # -------------------------
    point = data[4]
    ax.plot(point[0], point[1], "go", markersize=10)

    neighbors_0 = algo._get_neighbors(point, features=[0])
    neighbors_1 = algo._get_neighbors(point, features=[1])
    neighbors_01 = algo._get_neighbors(point, features=[0, 1])

    ax.plot(neighbors_0[:, 0], neighbors_0[:, 1], "bx", markersize=10)
    ax.plot(neighbors_1[:, 0], neighbors_1[:, 1], "gx", markersize=10)
    ax.plot(neighbors_01[:, 0], neighbors_01[:, 1], "go", markersize=15)
    ax.plot(point[0], point[1], "ro", markersize=10)

    algo._get_best_subspace(point)
    # -------------------------------------------------------------------


    # INITIALISE pq
    # -------------------------------------------------------------------
    pq = np.zeros((data.shape[0], 3))
    reach_dist_d1 = np.full(data.shape[0], fill_value=np.NaN)
    reach_dist_d2 = reach_dist_d1.copy()
    index = np.arange(0, data.shape[0])

    pq[:, 0] = index
    pq[:, 1] = reach_dist_d1
    pq[:, 2] = reach_dist_d2
    index = 0
    # -------------------------------------------------------------------

    # Compute Reachability by Walk
    # -------------------------------------------------------------------
    for index in range(pq.shape[0]):
        p_index = int(pq[index, 0])
        point = data[p_index]

        # calculate SDists
        # ----------
        d1, d2 = get_d(p_index, preference_vector)


        # get MU nearest neighbor of p
        # ----------
        # TODO: In RESPECT TO SDIST !!!!
        d1_value = np.argsort(d1)[mu]
        # p_mu = _get_mu_nearest_neighbor_index(point, data, features=w_p, mu=mu)

        # p_mu =


        # # Calculate ReachDist
        # # ----------
        # d1_mu = d1[p_mu]
        # d2_mu = d2[p_mu]
        # # if d1 are equal, d2 is MAXIMIZED
        # d1_same = (d1 == d1_mu)
        # d2[d1_same] = np.maximum(d2, d2_mu)[d1_same]
        #
        # # if d1 are not equal, take maximum d1
        # d1[~d1_same] = np.maximum(d1, d1_mu)[~d1_same]
        # d2[~d1_same] = np.maximum(d2, d2_mu)[~d1_same]

        # Update PQ
        # ----------
        # ADD new VALUES to d1 d2 of each point
        pq_sorted = pq[np.argsort(pq[:, 0])]
        pq_sorted = update_pq(pq_sorted, d1, d2)
        pq = pq_sorted[np.array(pq[:,0], dtype="int64")]

        # Sort pq
        # ----------
        # SORT PQs
        pq_sort = pq[index+1:]
        pq_sort = pq_sort[pq_sort[:, 2].argsort()]
        pq_sort = pq_sort[pq_sort[:, 1].argsort(kind='mergesort')]
        pq[index+1:] = pq_sort

        # PLOT ANIMATION
        # ---------------------------------------------------------------
        ax.plot(data[p_index, 0], data[p_index, 1], "go", markersize=20)
        for j in range(pq.shape[0]):
        #     ax.text(data[j, 0], data[j, 1] + 0.2, s=W_pq[j])
        #     ax.text(data[j, 0], data[j, 1] + 0.3, s=Lambda_pq[j])
        #     ax.text(data[j, 0], data[j, 1] + 0.4, s=is_included[j])
        #     ax.text(data[j, 0], data[j, 1] + 0.5, s=is_parallel[j])
        #     ax.text(data[j, 0], data[j, 1] + 0.2, s=d1[j])
        #     ax.text(data[j, 0], data[j, 1] + 0.4, s=d2[j])
            ax.text(data[j, 0], data[j, 1] + 0.4, s=pq[j, 1])
            ax.text(data[j, 0], data[j, 1] + 0.2, s=pq[j, 2])
        plt.pause(3)

        for nr in range(10):
            for txt in ax.texts:
                txt.set_visible(False)
                txt.remove()
        # ---------------------------------------------------------------

    fig, ax = plt.subplots()
    ax.plot(pq[:, 1], "bo")
    return True


def get_d(p_index, preference_vector):

    w_p = preference_vector[p_index]
    W_q = preference_vector
    W_pq = W_q * w_p
    Lambda_pq = W_pq.shape[1] - W_pq.sum(axis=1)

    is_included_p = (w_p == W_pq).all(axis=1)
    is_included_q = (preference_vector == W_pq).all(axis=1)
    is_included = is_included_q + is_included_p

    feature_matrix = W_pq
    is_parallel = DIST_v2(data[p_index], data, feature_matrix) > 2 * epsilon
    delta_pq = is_included * is_parallel

    W_inverse = ~W_pq
    d2 = DIST_v2(data[p_index], data, W_inverse)
    d1 = Lambda_pq + delta_pq
    return (d1, d2)


def _get_mu_nearest_neighbor_index(point, data, features, mu):
    # -----------------------
    if mu > len(data):
        raise ValueError("Mu is bigger than the total number of elements in the data."
                         "This is unreasonable. Please set it to a smaller (integer) value"
                         )
    # -----------------------
    dist = DIST(point[features], data[:, features])      # data[features] is a projection
    sorting_index = dist.argsort()
    return sorting_index[mu-1]




def update_pq(pq, d1, d2):
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


def DIST_v2(point, data, feature_matrix):
    dist_matrix = (point - data)**2
    projected_dist_matrix = np.multiply(dist_matrix, feature_matrix)
    final_dist_vector = np.sqrt(np.sum(projected_dist_matrix, axis=1))
    return final_dist_vector


def SDIST(data, preference_vector, row_p, row_q, epsilon=0.1):
    w_p = preference_vector[row_p]
    w_q = preference_vector[row_q]

    w_pq = w_p and w_q
    lambda_pq = w_pq.sum()

    is_included_p = (w_pq == w_p).all()
    is_included_q = (w_pq == w_q).all()

    features = [i for i, val in enumerate(w_pq) if True]
    is_parallel = (DIST(data[row_p][features], data[row_q][features][np.newaxis]) > 2 * epsilon).all()

    delta_pq = (is_included_p or is_included_q) and is_parallel

    features_inverse = [i for i, val in enumerate(w_pq) if False]
    d2 = DIST(data[row_p][features_inverse], data[row_q][features_inverse][np.newaxis])[0]
    d1 = lambda_pq + delta_pq
    return (d1, d2)


def DIST(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


if __name__ == '__main__':
    # Get Data
    # -------------------------------------------------------------------
    dataframe = pd.read_csv(fpath, sep=" ", comment="#", header=None)
    data = dataframe.values

    data = np.array([
        [1, 6],
        [2, 6],
        [3, 6],
        [4, 6],
        [5, 6],
        [1, 3], # CLUSTER 2
        [2, 3],
        [3, 3],
        [4, 3],
        [5, 3],
        [6, 6], # CLUSTER 3
        [6, 7],
        [6, 8],
        [6, 5],
        [6, 4],
        [6, 3],
    ])
    # -------------------------------------------------------------------

    # Algo
    # -------------------------------------------------------------------

    mu = 3
    epsilon = 0.1

    dish(data=data, mu=mu, epsilon=epsilon)
    # -------------------------------------------------------------------

# def get_nearest_neighbors(point, data, features=[], epsilon=0.1):
#     if len(features) < 1:
#         raise ValueError("Dimension of the features has to be at least 1")
#     is_relevant = DIST(point[features], data[:, features]) < epsilon
#     relevant_data = data[is_relevant]
#     return relevant_data
#
#
# def get_nr_of_neighbors(point, data, epsilon=0.1):
#     nr_of_neighbors = np.zeros(data.shape[1])
#     for feature in range(data.shape[1]):
#         nr_of_neighbors[feature] = get_nearest_neighbors(point, data, features=[feature], epsilon=epsilon).shape[0]
#     return nr_of_neighbors
#
#
# def get_best_subspace(point, data, nr_of_neighbors, mu=10, epsilon=0.1):
#     best_subspace = [nr_of_neighbors.argmax()]
#     is_candidate = (nr_of_neighbors > mu)
#     candidates = [i for i, val in enumerate(is_candidate) if True]
#     if len(candidates) >= 1:
#         for candidate in candidates:
#             new_features = list(set(best_subspace + [candidate]))
#             if get_nearest_neighbors(point, data, features=new_features, epsilon=epsilon).shape[0] > mu:
#                 best_subspace = new_features
#     return best_subspace
#
