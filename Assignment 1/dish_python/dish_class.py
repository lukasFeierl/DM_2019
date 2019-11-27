import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DiSH():
    """
    Initiates the DiSH Algorithm for clustering data.
    SOURCE: TODO
    """
    def __init__(self, epsilon, mu):
        self.epsilon = epsilon          # INPUT
        self.mu = mu
        self.data = None
        self.cluster_list = None        # Output
        self.pq = None
        return

    #%% Main
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, data, SHOW_PLOT=True):
        """ uses the data to find cluster hierachies in it. returns found clusters."""
        self.data = data

        preference_vector = self._get_preference_vectors()        # w(p) as array for each row (i.e. point) of the data
        pq = self._calculate_pq(preference_vector)                # Columns: [index_point, d1_point, d2_point]

        cluster_list = self._extract_cluster(pq, preference_vector)
        self.__save_and_show_results(pq=pq, cluster_list=cluster_list, SHOW_PLOT=SHOW_PLOT)
        return cluster_list

    #%% PREFERENCE VECTOR CALCULATION
    # ------------------------------------------------------------------------------------------------------------------

    def _get_preference_vectors(self):
        """ calculates the preference vector w(p) of each point, assigning "True" to each feature which is in the "best
        subspace" - thus relevant for clustering.
        The number of zero-values in the preference vector is the subspace dimensionality.
        """
        preference_vector = np.zeros(self.data.shape, dtype=bool)       # initialising - stores w(p) for each point
        for point_index, point in enumerate(self.data):
            is_best_subspace = self._get_best_subspace(point)          # calculates which attributes are relevant
            preference_vector[point_index][is_best_subspace] = True
        return preference_vector

    def _get_best_subspace(self, point):
        """ calculates the best subspace for a given point. An attribute contributes to the subspace if enough
        elements are in the neighborhood of the point in a radius epsilon. """
        nr_neighbors_per_dim = self._get_neighbor_count(point)
        candidate_features = self._get_candidate_attributes(nr_neighbors_per_dim)
        best_subspace = self._best_first_search(point, nr_neighbors_per_dim, candidate_features)
        return best_subspace

    def _get_neighbor_count(self, point):
        """ counts how many datapoints are in the neighborhood of the point, if projected along each feature of
        the data. Returns a vector for each row of the data."""
        nr_of_features = point.shape[0]
        neighbor_count = np.zeros(nr_of_features)
        for feature in range(nr_of_features):
            neighbors = self._get_neighbors(point, features=[feature])
            neighbor_count[feature] = len(neighbors)
        return neighbor_count

    def _get_neighbors(self, point, features):
        """ returns all neighbors of a point, if projected along features in a radius epsilon"""
        data_projected = self.data[:, features]
        point_projected = point[features]
        is_near = self._DIST(point_projected, data_projected) <= self.epsilon      # data[features] is a projection
        return self.data[is_near]

    def _DIST(self, point, data):
        """ calculates the euclidean distance between point and all points in data"""
        return np.sqrt(np.sum((point - data)**2, axis=1))

    def _get_candidate_attributes(self, nr_of_neighbors_per_dim):
        is_candidate = (nr_of_neighbors_per_dim >= self.mu)
        candidate_features = [i for i, val in enumerate(is_candidate) if val == True]        # returns index of TRUE elements
        return candidate_features

    def _best_first_search(self, point, nr_neighbors_per_dim, candidate_features):
        """ approach to find best subspace. Takes the feature with the most neighbors first, then merges dimension and
        test if criterion still holds."""
        nr_neighbors_per_dim = nr_neighbors_per_dim.copy()
        best_subspace = []
        for index in range(len(candidate_features)):
            best_feature = nr_neighbors_per_dim.argmax()
            nr_neighbors_per_dim[best_feature] = -1                 # set "visited" features to "already searched"
            proposed_combination = best_subspace + [best_feature]
            if len(self._get_neighbors(point, features=proposed_combination)) >= self.mu:
                best_subspace = proposed_combination
        return best_subspace


    #%% REACHABILITY DISTANCE CALCULATION
    # ------------------------------------------------------------------------------------------------------------------

    def _calculate_pq(self, preference_vector):
        """ calculates pq for each point in data, where the column 0 saves the index of the point, and column 1 and 2
        saves the d1 and d2 subspace distance of the points. The assignment is done by a WALK through the data """
        pq = self._initialise_pq()

        # fig, ax = plt.subplots()
        # ax.plot(self.data[:, 0], self.data[:, 1], "ko")
        # fig.show()

        for index in range(pq.shape[0]):
            p_index = int(pq[index, 0])                                      # index of point (pq is ordered by RDist)
            d1, d2 = self._get_subspace_distance(p_index, preference_vector)
            d1, d2 = self._get_reachability_distance(d1, d2)
            pq = self._update_pq(pq, d1, d2)
            pq = self._resort_pq(pq, index)
            # print(pq[index+1, 1:])
            # ax.plot(self.data[p_index, 0], self.data[p_index, 1], "ro")

        return pq

    def _initialise_pq(self):
        """ initialises pq matrix. Each row belongs to one point. Three columns with index of the points, and d1 and
        d2 values."""
        pq_init = np.full((self.data.shape[0], 3), fill_value=np.NaN)
        index_init = np.arange(0, self.data.shape[0])
        pq_init[:, 0] = index_init
        return pq_init

    def _get_subspace_distance(self, p_index, preference_vector):
        """ calculates the subspace distance between the point and all the other points in the data.
        The subspace distance consists of two values d1 and d2. d1 represents the dimensionality in respect to another
        point. Two points have the same d1 if they are k.dim clusters belonging to the same cluster, or k-1 dim. clusters
        belonging to separate clusters (i.e not the same preference vector or to far away from each other).
        """
        w_p = preference_vector[p_index]
        W_q = preference_vector
        W_pq = W_q * w_p
        dimensionality_pq = W_pq.shape[1] - W_pq.sum(axis=1)        # called lambda in the paper

        is_included_p = (w_p == W_pq).all(axis=1)                   # meaning they have similar preference vectors
        is_included_q = (preference_vector == W_pq).all(axis=1)
        is_included = is_included_q + is_included_p

        point = self.data[p_index]
        is_parallel = self._DIST_projected(point, self.data, preference_matrix=W_pq) > 2*self.epsilon
        delta_pq = is_included * is_parallel

        d1 = dimensionality_pq + delta_pq           # measuring the "dimensionality" of the two points combined.
        W_inverse = ~W_pq
        d2 = self._DIST_projected(point, self.data, W_inverse)    # measuring the distance inside the combined cluster.
        return d1, d2

    def _DIST_projected(self, point, data, preference_matrix):
        """ calculates the euclidean distance, but uses the preference_vectors to project the data to lower dimension"""
        dist_matrix = (point - data) ** 2
        projected_dist_matrix = np.multiply(dist_matrix, preference_matrix)
        if len(data.shape) == 1:
            final_dist_vector = np.sqrt(np.sum(projected_dist_matrix))
        else:
            final_dist_vector = np.sqrt(np.sum(projected_dist_matrix, axis=1))
        return final_dist_vector

    def _get_reachability_distance(self, d1, d2):
        """ to avoid single link effect, the sdist of the mu nearest neighbor of the point in respect to p is used
        as minimum sdist. If the point is in a cluster with less then mu neighbors this then results in beeing a
        one-point cluster (hence, no cluster) """
        d2_orig = d2.copy()
        d = np.vstack((d1, d2)).T
        d_argsort = np.lexsort((d[:, 1], d[:, 0]))  # Sort according to d1, then d2
        d_sorted = d[d_argsort]
        d_mu = d_sorted[self.mu]
        d_sorted[:self.mu] = d_mu       # is equal to max(sdist(p, r), sdist(p, mu))
        d[d_argsort] = d_sorted

        return d[:, 0], d[:, 1]

    def _update_pq(self, pq, d1, d2):
        """ re-assigns new d1 and d2 values to pq"""
        pq_sorted = pq[np.argsort(pq[:, 0])]            # resort by point_index, beacuse d1 & d2 are sorted this way too
        pq_sorted = self._update_sorted_pq(pq_sorted, d1, d2)
        pq = pq_sorted[np.array(pq[:, 0], dtype="int64")]
        return pq

    def _update_sorted_pq(self, pq, d1, d2):
        """ re-assigns new d1 and d2 values to pq (which is sorted by point_index)"""
        d1_old_new = np.vstack((pq[:, 1], d1))          # [0,:] <- old values     [1,:] <- new values
        d2_old_new = np.vstack((pq[:, 2], d2))

        # if d1 are equal, the minimum d2 us used
        d1_is_equal = (d1_old_new[0] == d1_old_new[1])
        pq[:, 2][d1_is_equal] = np.nanmin(d2_old_new, axis=0)[d1_is_equal]

        # if d1 are unequal, d1 and d2 are taken from the one where d1 is smaller.
        is_minimum = np.zeros(d1_old_new.shape, dtype=bool)
        minimum_value = np.nanmin(d1_old_new, axis=0)
        is_minimum[d1_old_new == minimum_value] = True
        is_minimum[:, d1_is_equal] = False
        pq[:, 1][~d1_is_equal] = d1_old_new[is_minimum]
        pq[:, 2][~d1_is_equal] = d2_old_new[is_minimum]

        return pq

    def _resort_pq(self, pq, index):
        pq_sort = pq[index + 1:]                                    # only sort not yet visited points !!
        pq_argsort = np.lexsort((pq_sort[:, 2], pq_sort[:, 1]))     # Sort according to d1, then d2
        pq[index + 1:] = pq_sort[pq_argsort]                        # Apply reordering
        return pq

    #%% EXTRACTING CLUSTERS
    # ------------------------------------------------------------------------------------------------------------------

    def _extract_cluster(self, pq, preference_vector):
        """ extracts the cluster based on the pq which now includes the d1 and d2 values from the walk through the
        cluster. The walk - always using points that are closest to visited ones - enables easy extraction."""
        cluster_order = pq      # cols: [point_index, d1, d2]
        cluster_list = []       # list with individual clusters (containing numpy arrays with all points of the cluster)

        predecessor = cluster_order[0]      # first point / previous point

        for object in cluster_order:
            o_index = int(object[0])                # object is a point with [index, d1 and d2]
            p_index = int(predecessor[0])

            point_o = self.data[o_index]

            w_o = preference_vector[o_index]
            w_p = preference_vector[p_index]
            w_op = w_p*w_o

            # plt.plot(point_o[0], point_o[1], 'ko')
            # plt.pause(0.1)

            # Get corresponding cluster
            # ---------------------------
            corresponding_cluster = None
            for cluster in cluster_list:
                c_center = cluster["data"].mean(axis=0)

                has_same_preference_vector = (cluster["w_c"] == w_op).all()
                is_near_enough = self._DIST_projected(point_o, c_center, preference_matrix=w_op) <= 2*self.epsilon

                if has_same_preference_vector and is_near_enough:
                    # print("existing cluster found")
                    corresponding_cluster = cluster
                    cluster["data"] = np.vstack((cluster["data"], point_o))
                    break

            if corresponding_cluster is None:
                # print("new cluster")
                # plt.plot(point_o[0], point_o[1], 'ro')
                cluster_data = np.array(point_o)[np.newaxis]
                cluster_w_c = w_o
                cluster_list += [{"data": cluster_data,
                                  "w_c": cluster_w_c}]
            predecessor = object

        return cluster_list

    #%% DISPLAY
    # ------------------------------------------------------------------------------------------------------------------

    def __save_and_show_results(self, pq, cluster_list, SHOW_PLOT=True):
        self.pq = pq
        self.cluster_list = cluster_list
        print(self)
        if SHOW_PLOT:
            self.plot_cluster(cluster_list)

    def __str__(self):
        return "# ----------------------------------------------------------\n" \
               " DiSH-Algorithm: \n"+"\tmu:\t\t\t"+str(self.mu)+"\n\tepsilon:\t"+str(self.epsilon)+"" \
               "\n# ----------------------------------------------------------"

    #%% PLOT
    # ------------------------------------------------------------------------------------------------------------------

    def plot_reference_vectors(self):
        preference_vector = self._get_preference_vectors()
        pref_ax0 = self.data[preference_vector[:, 0]]
        pref_ax1 = self.data[preference_vector[:, 1]]

        fig, ax = plt.subplots()
        ax.plot(self.data[:, 0], self.data[:, 1], "ko", markersize=10)
        ax.plot(pref_ax0[:, 0], pref_ax0[:, 1], "r^", label="w_up == True", markersize=10)
        ax.plot(pref_ax1[:, 0], pref_ax1[:, 1], "y>", label="w_right == True", markersize=10)
        return fig, ax

    def plot_reachablity_plot(self):
        fig, ax = plt.subplots()
        fig.suptitle("Reachability Plot")
        ax.plot(self.pq[:, 1], 'o', color="black", label="RDist")
        ax.fill_between(np.arange(0, self.pq.shape[0]), self.pq[:, 1], 0, color="black")
        ax.legend(loc="upper center")
        return fig, ax

    def plot_cluster(self, cluster_list):
        fig, ax = plt.subplots()
        fig.suptitle("DiSH Clustering Results")
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for i, cluster in enumerate(cluster_list):
            cluster_data = cluster["data"]
            cluster_center = cluster["data"].mean(axis=0)

            if cluster_data.shape[0] < self.mu:
                ax.plot(cluster_data[:, 0], cluster_data[:, 1], 'ko')
            else:
                ax.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', color=colors[i % 9],
                        label="Cluster " + str(cluster["w_c"]))
                ax.plot(cluster_center[0], cluster_center[1], 'x', markersize=20, color=colors[i % 9])

        ax.legend(loc="upper right")
        return fig, ax


if __name__ == '__main__':

    # Get Data
    # -------------------------------------------------------------------
    fpath = r"../datasets/mouse.csv"
    dataframe = pd.read_csv(fpath, sep=" ", comment="#", header=None)
    data = dataframe.values
    # -------------------------------------------------------------------

    # FIT
    # -------------------------------------------------------------------
    self = DiSH(epsilon=0.1, mu=40)
    self.fit(data)
    # -------------------------------------------------------------------
