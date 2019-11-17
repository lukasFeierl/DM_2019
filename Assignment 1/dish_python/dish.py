"""
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dish(data, mu=4, epsilon=0.1):
    DO_PLOT = True
    from dish_class import DiSH
    self = DiSH(epsilon=epsilon, mu=mu)

    # -------------------------------------------------------------------
    self.data = data
    self.nr_of_features = 2

    # ALGO
    # -------------------------------------------------------------------
    preference_vector = self._get_preference_vectors()
    pq = self.init_pq()

    for index in range(pq.shape[0]):
        p_index = int(pq[index, 0])
        d1, d2 = self.get_subspace_distance(p_index, preference_vector)

        # TODO:  get MU nearest neighbor of p in respect to SDIST
        pq = self.update_pq(pq, d1, d2)
        pq = self.sort_pq(pq, index)
        if DO_PLOT:
            # PLOT ANIMATION
            # ---------------------------------------------------------------
            # point = data[p_index]
            # ax.plot(data[p_index, 0], data[p_index, 1], "go", markersize=10)
            # ax.plot(data[mu_index, 0], data[mu_index, 1], "rx", markersize=20)
            # for j in range(pq.shape[0]):
            #     ax.text(data[j, 0], data[j, 1] + 0.012, s="w(p,q): "+str(W_pq[j]))
            #     ax.text(data[j, 0], data[j, 1] + 0.006, s=u"{\lambda}(p,q): "+str(Lambda_pq[j]))
            #     ax.text(data[j, 0], data[j, 1] + 0.007, s="included: "+str(is_included[j]))
            #     ax.text(data[j, 0], data[j, 1] + 0.010, s="parallel: "+str(is_parallel[j]))
            #     ax.text(data[j, 0], data[j, 1] + 0.002, s="d1: "+str(d1[j]))
            #     ax.text(data[j, 0], data[j, 1] + 0.004, s="d2: "+str(d2[j]))
            # plt.pause(0.2)
            #
            # for nr in range(10):
            #     for txt in ax.texts:
            #         txt.set_visible(False)
            #         txt.remove()
            # # ---------------------------------------------------------------
            pass

    cluster_list = self.extract_cluster(pq, preference_vector)
    # -------------------------------------------------------------------

    cluster_list = self.fit(data=data)

    if DO_PLOT:
        plot_reference_vectors(data, preference_vector)
        plot_cluster(cluster_list)
        plot_reachablity_plot(pq)

    print(self)
    return cluster_list


def plot_reference_vectors(data, preference_vector):
    pref_ax0 = data[preference_vector[:, 0]]
    pref_ax1 = data[preference_vector[:, 1]]

    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], "ko", markersize=10)
    ax.plot(pref_ax0[:, 0], pref_ax0[:, 1], "r^", label="w_up == True", markersize=10)
    ax.plot(pref_ax1[:, 0], pref_ax1[:, 1], "y>", label="w_right == True", markersize=10)
    return fig, ax


# def plot_neighbors(data, point, algo, ax):
#     if ax is None:
#         fig, ax = plt.subplots()
#         ax.plot(data[:, 0], data[:, 1], "ko", markersize=10)
#
#     ax.plot(point[0], point[1], "go", markersize=10)
#     neighbors_0 = algo._get_neighbors(point, features=[0])
#     neighbors_1 = algo._get_neighbors(point, features=[1])
#     neighbors_01 = algo._get_neighbors(point, features=[0, 1])
#
#     ax.plot(neighbors_0[:, 0], neighbors_0[:, 1], "bx", markersize=10)
#     ax.plot(neighbors_1[:, 0], neighbors_1[:, 1], "gx", markersize=10)
#     ax.plot(neighbors_01[:, 0], neighbors_01[:, 1], "go", markersize=15)
#     ax.plot(point[0], point[1], "ro", markersize=10)
#     return fig, ax


def plot_data(data):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], "ko", markersize=10)
    return fig, ax



def plot_reachablity_plot(pq):
    fig, ax = plt.subplots()
    fig.suptitle("Reachability Plot")
    ax.plot(pq[:, 1], 'o', color="black")
    ax.fill_between(np.arange(0, pq.shape[0]), pq[:, 1], 0, color="black")
    return fig, ax


def plot_cluster(cluster_list):
    fig, ax = plt.subplots()
    fig.suptitle("DiSH Clustering Results")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, cluster in enumerate(cluster_list):
        cluster_data = cluster["data"]
        cluster_center = cluster["data"].mean(axis=0)

        if cluster_data.shape[0] < 2:
            ax.plot(cluster_data[:, 0], cluster_data[:, 1], 'ko')
        else:
            ax.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', color=colors[i % 9],
                    label="Cluster " + str(cluster["w_c"]))
            ax.plot(cluster_center[0], cluster_center[1], 'x', markersize=20, color=colors[i % 9])

    ax.legend(loc="upper right")
    return fig, ax


def init_pq(data):
    pq = np.zeros((data.shape[0], 3))
    reach_dist_d1 = np.full(data.shape[0], fill_value=np.NaN)
    reach_dist_d2 = reach_dist_d1.copy()
    index = np.arange(0, data.shape[0])

    pq[:, 0] = index
    pq[:, 1] = reach_dist_d1
    pq[:, 2] = reach_dist_d2
    return pq


def update_pq(pq, d1, d2):
    # ADD new VALUES to d1 d2 of each point
    pq_sorted = pq[np.argsort(pq[:, 0])]
    pq_sorted = update_sorted_pq(pq_sorted, d1, d2)
    pq = pq_sorted[np.array(pq[:, 0], dtype="int64")]
    return pq


def sort_pq(pq, index):
    pq_sort = pq[index + 1:]
    pq_argsort = np.lexsort((pq_sort[:, 2], pq_sort[:, 1]))
    pq[index + 1:] = pq_sort[pq_argsort]
    return pq


def get_wpq(p_index, preference_vector):
    w_p = preference_vector[p_index]
    W_q = preference_vector
    W_pq = W_q * w_p
    return W_pq


def calculate_SDist(p_index, preference_vector, epsilon, data):
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


def update_sorted_pq(pq, d1, d2):
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
    dist_matrix = (point - data) ** 2
    projected_dist_matrix = np.multiply(dist_matrix, feature_matrix)
    if len(data.shape) == 1:
        final_dist_vector = np.sqrt(np.sum(projected_dist_matrix))
    else:
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
    if len(data.shape) > 1:
        return np.sqrt(np.sum((point - data) ** 2, axis=1))
    else:
        return np.sqrt(np.sum((point - data)**2))


def extract_cluster(pq, preference_vector, data, epsilon):

    cluster_order = pq      # point_index, d1, d2
    cluster_list = []       # list with individual clusters (containing numpy arrays with all points of the cluster)

    predecessor = cluster_order[0]

    for object in cluster_order:
        o_index = int(object[0])
        p_index = int(predecessor[0])

        w_o = preference_vector[o_index]
        w_p = preference_vector[p_index]
        w_op = w_p*w_o

        corresponding_cluster = None

        for cluster in cluster_list:
            c_center = cluster["data"].mean(axis=0)
            if (cluster["w_c"] == w_op).all() and (DIST_v2(data[o_index], c_center, feature_matrix=w_op) <= 2*epsilon):
                print("existing cluster found")
                corresponding_cluster = cluster
                cluster["data"] = np.vstack((cluster["data"], data[o_index]))
                break

        if corresponding_cluster is None:
            print("new cluster")
            cluster_data = np.array(data[o_index])[np.newaxis]
            cluster_w_c = w_o
            cluster_list += [{"data": cluster_data,
                              "w_c": cluster_w_c}]
        predecessor = object

    return cluster_list


if __name__ == '__main__':
    fpath = r"../datasets/mouse.csv"
    # fpath = r"../datasets/simple_lines.csv"

    # Get Data
    # -------------------------------------------------------------------
    dataframe = pd.read_csv(fpath, sep=" ", comment="#", header=None)
    data = dataframe.values
    # -------------------------------------------------------------------

    # Algo
    # -------------------------------------------------------------------
    mu = 40
    epsilon = 0.1
    dish(data=data, mu=mu, epsilon=epsilon)
    # -------------------------------------------------------------------
