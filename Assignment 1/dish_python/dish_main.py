import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def DIST(point, data):
    """ Calculates the euclidean distance between point and all points in data.

    Parameters:
    -----------
    point : Vector of a single point.
    data : Matrix of all data points.

    Returns:
    --------
    Matrix containing distances from all points to the input point."""
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def get_neighbors(point, data, features, epsilon):
    """ Returns all neighbors of a point, if projected along features in a radius epsilon.

    Parameters:
    -----------
    point : Vector of a single point.
    data : Matrix of all data points.
    features : List of features along which the points are projected.
    epsilon : Distance within which points are considered close to each other.

    Returns:
    --------
    Matrix of data points within the epsilon range of the point.
    """
    is_near = DIST(data[:, features], point[features]) <= epsilon         # point[features] = projection to one feature
    return data[is_near]


def get_best_subspace(point, data, mu, epsilon, TEST_PLOT=False):
    """ calculates the best subspace for a given point. An attribute/feature contributes to the best subspace
	if enough elements are in the neighborhood
    of the point in a radius epsilon.

    Parameters:
    -----------
    point : Vector of a single point.
    data : Matrix of all data points.
    mu : Minimum numbers of points within the epsilon range.
    epsilon : Distance within which points are considered close to each other.

    Returns:
    --------
    List of best subspaces for the given point.
    """
    #-----------------------------
    if TEST_PLOT:
        fig, ax = plt.subplots()
        fig.suptitle("DiSH - get_best_subspace()")
        ax.plot(data[:,0], data[:,1], "k.")
        ax.plot(point[0], point[1], "ro")
    #-----------------------------

    # initialising
    nr_of_dims = point.shape[0]                 # feature dimensions

    # count neighbors in each (1D) feature dimension
    neighbor_count = np.zeros(nr_of_dims)
    for feature in range(nr_of_dims):
        neighbors = get_neighbors(point, data=data, features=[feature], epsilon=epsilon)
        neighbor_count[feature] = len(neighbors)

        # -----------------------------
        if TEST_PLOT:
            ax.plot(neighbors[:,0], neighbors[:,1], ".", label="feature: "+ str(feature)+" (nr="+str(len(neighbors))+")")
        #-----------------------------

    # get all candidate attributes
    is_candidate = (neighbor_count >= mu)
    candidate_features = [i for i, value in enumerate(is_candidate) if value == True]

    # Apply best-First_search to find the best subspace
    best_subspace = []
    for index in range(len(candidate_features)):

        # Propose a new subspace
        best_feature = neighbor_count.argmax()                  # get feature with highest neighbor count
        neighbor_count[best_feature] = -1                       # set current features to "visited/already tried"
        proposed_featurespace = best_subspace + [best_feature]  # add feature and subspace to form new proposed-subspace

        # Check if criterion holds true for proposed subspace
        neighbors = get_neighbors(point, data=data, features=proposed_featurespace, epsilon=epsilon)
        if len(neighbors) >= mu:
            best_subspace = proposed_featurespace

    # -----------------------------
    if TEST_PLOT:
        neighbors = get_neighbors(point, data=data, features=best_subspace, epsilon=epsilon)
        ax.plot(neighbors[:, 0], neighbors[:, 1], '.', label="best subspace: "+ str(best_subspace)+" (nr="+str(len(neighbors))+")")
        ax.legend()
    # -----------------------------

    return best_subspace


def get_preference_vectors(data, mu, epsilon):
    """ Calculates the preference vector w(p) of each point, assigning "True" to each feature which is in the "best subspace"
	- thus relevant for clustering - and "False" otherwise. The number of False-values in the preference vector is the
	called the subspace dimensionality Lambda.

    Parameters:
    -----------
    data : Matrix of all data points.
    mu : Minimum numbers of points within the epsilon range.
    epsilon : Distance within which points are considered close to each other.

    Returns:
    --------
    Vector with Boolean entries for each feature (True if feature is among the list of best subspaces and False otherwise).

    """
    # initialising - stores w(p) for each point
    preference_vector = np.zeros(data.shape, dtype=bool)

    for point_index, point in enumerate(data):
        # calculate which features are relevant
        is_best_subspace = get_best_subspace(point, data, mu=mu, epsilon=epsilon)
        preference_vector[point_index][is_best_subspace] = True

    return preference_vector


def DIST_projected(point, data, preference_matrix):
    """ Calculates the Euclidean distance, but uses the preference_matrix to project the data to lower dimensions.
   
    Parameters:  
    -----------
    point : Vector of a single point.
    data : Matrix of all data points.
    preference_matrix : Matrix containing preference vectors for each point.

    Returns: 
    --------
    Matrix containing distances p-q for each point q in the data.
    """
    dist_matrix = (point - data) ** 2

    # Project to feature dimensions specified in preference Matrix
    projected_dist_matrix = np.multiply(dist_matrix, preference_matrix)

    if len(data.shape) == 1:
        dist_vector = np.sqrt(np.sum(projected_dist_matrix))
    else:
        dist_vector = np.sqrt(np.sum(projected_dist_matrix, axis=1))

    return dist_vector


def get_subspace_distance(data, p_index, preference_vector, epsilon):
    """ Calculates the subspace distance between the point and all the other points in the data.
    The subspace distance consists of two values d1 and d2. d1 represents the dimensionality in respect to another
    point. Two points have the same d1 if they are k.dim clusters belonging to the same cluster, or k-1 dim. clusters
    belonging to separate clusters (i.e not the same preference vector or too far away from each other).

    Parameters:
    -----------
    data : Matrix of all data points.
    p_index : Index of the given point.
    preference_vector : Matrix containing preference vectors for each point.
    epsilon : Distance within which points are considered close to each other.

    Returns:
    --------
    Two values d1 and d2 for the given point.
    """
    # Calculate dimensionality Lambda (Part one of d1)
    w_p = preference_vector[p_index]
    W_q = preference_vector
    W_pq = W_q * w_p
    dimensionality_pq = W_pq.shape[1] - W_pq.sum(axis=1)        # called lambda in the paper

    # Calculated Detla pq (PArt two of d1)
    is_included_p = (w_p == W_pq).all(axis=1)                   # meaning they have similar preference vectors
    is_included_q = (preference_vector == W_pq).all(axis=1)
    is_included = is_included_q + is_included_p

    point = data[p_index]
    is_parallel = DIST_projected(point, data, preference_matrix=W_pq) > 2*epsilon
    delta_pq = is_included * is_parallel

    # Set D1
    d1 = dimensionality_pq + delta_pq  # measuring the "dimensionality" of the two points combined.

    # Calculate d2
    W_inverse = ~W_pq
    d2 = DIST_projected(point, data, W_inverse)    # measuring the distance inside the combined cluster.

    return d1, d2


def get_reachability_distance(d1, d2, mu):
    """ To avoid single linkage effects, the SDIST of the mu-nearest-neighbor of the point with respect to p is used
        as minimum SDIST. If the point is in a cluster with less then mu neighbors, this then results in a
        one-point cluster (hence, no cluster).

    Parameters:
    -----------
    d1 : d1-value of a given point.
    d2 : d2-value of a given point.
    mu : Minimum numbers of neighbors.

    Returns:
    --------
    Reachability distances (two values)? for the given point.
    """
    d = np.vstack((d1, d2)).T
    d_argsort = np.lexsort((d[:, 1], d[:, 0]))  # Sort according to d1, then d2
    d_sorted = d[d_argsort]
    d_mu = d_sorted[mu]
    d_sorted[:mu] = d_mu       # is equal to max(sdist(p, r), sdist(p, mu))
    d[d_argsort] = d_sorted

    return d[:, 0], d[:, 1]


def update_pq(pq, d1, d2):
    """ To avoid single linkage effects, the SDIST of the mu-nearest-neighbor of the point with respect to p is used
    as minimum SDIST. If the point is in a cluster with less then mu neighbors, this then results in a
    one-point cluster (hence, no cluster).

    Parameters:
    -----------
    pq : Priority queue.
    d1 : d1-value of a given point.
    d2 : d2-value of a given point.

    Returns:
    --------
    Reordered priority queue pq according to the reachability distances.
    """
    pq = pq.copy()
    d1_old_new = np.vstack((pq[:, 1], d1))  # [0,:] <- old values     [1,:] <- new values
    d2_old_new = np.vstack((pq[:, 2], d2))

    # if d1 are equal, the minimum d2 us used
    d1_is_equal = (d1_old_new[0] == d1_old_new[1])
    pq[:, 2][d1_is_equal] = np.nanmin(d2_old_new, axis=0)[d1_is_equal]

    # if d1 are unequal, d1 and d2 are taken from the one where d1 is smaller.
    argminimum = np.nanargmin(d1_old_new, axis=0)

    index_1 = argminimum[np.newaxis]
    index_2 = np.arange(pq.shape[0])
    pq[:, 1][~d1_is_equal] = d1_old_new[index_1, index_2][0][~d1_is_equal]
    pq[:, 2][~d1_is_equal] = d2_old_new[index_1, index_2][0][~d1_is_equal]

    return pq


def get_pq(data, preference_vector, epsilon, mu, PLOT_TEST=False):
    """ Determines the priority queue which is the final cluster order from which the clusters can be extracted.

        Parameters:
        -----------
        data : Matrix of all data points.
        preference_vector : Matrix containing preference vectors for each point.
        epsilon : Distance within which points are considered close to each other.
        mu : Minimum numbers of points within the epsilon range.

        Returns:
        --------
        Priority queue containing entries consisting of [index, d1, d2] which is the final cluster order.
    """
    # Initialise pq
    # -----------------------------
    # First column are indices from the points, second column are the d1-values, last column are the d2-values
    # d1, d2 initially infinity ..
    pq = np.full((data.shape[0], 3), fill_value=np.NaN)
    pq[:, 0] = np.arange(0, data.shape[0])

    if PLOT_TEST:
        fig, ax = plt.subplots()
        ax.plot(data[0], data[1], 'k.')

    # Calculate Distances
    # -----------------------------
    for index in range(data.shape[0]):

        # Calculate d1 and d2 (Subspace Distance)
        p_index = int(pq[index, 0])  # index of point (pq is ordered by RDist)
        d1, d2 = get_subspace_distance(data, p_index, preference_vector, epsilon=epsilon)

        # Calculate d1 and d2 (Reachablity Distance)
        d1, d2 = get_reachability_distance(d1, d2, mu=mu)

        # Assign new Values to PQ
        pq_sorted = pq[np.argsort(pq[:, 0])]  # resort by point_index (because d1 & d2 are sorted this way too)
        pq_sorted = update_pq(pq_sorted, d1, d2)
        pq = pq_sorted[np.array(pq[:, 0], dtype="int64")]

        # Reorder pq according to values
        shift_index = 1
        pq_sort = pq[index + shift_index:]  # only sort not yet visited points !!
        pq_argsort = np.lexsort((pq_sort[:, 2], pq_sort[:, 1]))  # Sort according to d1, then d2
        pq[index + shift_index:] = pq_sort[pq_argsort]  # Apply reordering

        if PLOT_TEST:
            d1 = pq[index:, 1]
            d2 = pq[index:, 2]
            plot_sdist(data[p_index], data[index:], sdist_p=(d1, d2), fig=fig, ax=ax)
            plt.pause(0.1)

    return pq


def extract_cluster(cluster_order, data, preference_vector, epsilon):
    """ Walks through the previously obtained cluster order (=priority queue) to extract the clusters.

    Parameters:
    -----------
    cluster_order : Priority queue containing entries consisting of [index, d1, d2].

    Returns:
    --------
    A dictionary containing the clusters and their points.
    """
    cluster_list = []  # list with individual clusters (containing numpy arrays with all points of the cluster)
    cluster_found = False
    predecessor = cluster_order[0]  # predecessor of current point

    for object in cluster_order:
        o_index = int(object[0])  # object is a point with [index, d1 and d2]
        p_index = int(predecessor[0])

        point_o = data[o_index]

        w_o = preference_vector[o_index]
        w_p = preference_vector[p_index]
        w_op = w_p * w_o

        # Get corresponding cluster
        # ---------------------------
        for cluster in cluster_list:
            c_center = cluster["data"].mean(axis=0)

            has_same_preference_vector = (cluster["w_c"] == w_op).all()
            is_near_enough = DIST_projected(point_o, c_center, preference_matrix=w_op) <= 2 * epsilon

            if has_same_preference_vector and is_near_enough:
                cluster_found = True
                cluster["data"] = np.vstack((cluster["data"], point_o))
                break

        if not cluster_found:
            print("Cluster " + str(len(cluster_list) + 1) + " found.")
            cluster_list += [{"data": np.array(point_o)[np.newaxis],
                              "w_c": w_o,
                              "num": len(cluster_list)+1,
                              },]

        # Set for next iteration
        cluster_found = False
        predecessor = object
    return cluster_list


def dish(data, epsilon, mu):
    """ Starts the DiSH Algorithm

    Parameters:
    -----------
    data : Matrix of all data points.
    epsilon : Distance within which points are considered close to each other.
    mu : Minimum numbers of points within the epsilon range.

    Returns:
    --------
    A dictionary containing the clusters and their points.
    """
    # Calculate Preference Vectors
    preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)  # w(p) as array for each row (i.e. point) of the data

    # Calculate ReachDistances
    pq = get_pq(data, preference_vector, epsilon=epsilon, mu=mu)

    # Extract Cluster
    cluster_list = extract_cluster(cluster_order=pq, data=data, preference_vector=preference_vector, epsilon=epsilon)

    return cluster_list


def build_hirarchy(cluster_list, epsilon, mu, PLOT_RESULTS=True):
    """ creates a hirarchy between the clusters using the dimensionality of the clusters combined with
    their distances to each other. As a result, low-dimensional clusters are assigned to high dimensional
    ones if their centers are near enough.

    Parameters:
    -----------
    cluster_list : Dictionary containing the clusters and their points.
    epsilon : Distance within which points are considered close to each other.
    mu : Minimum numbers of points within the epsilon range.

    Returns:
    -----------
    updated cluster_list where clusters with too less datapoins are discarded while valid clusters are assigned to at
    least one parent (with "is_child_of" attribute).
    """
    final_clusters = cluster_list.copy()
    dimensionality = cluster_list[0]["w_c"].shape[0]

    noise = {
        "lambda": 0,
        "w_c": np.array([False, False]),
        "data": np.array([], dtype=np.float).reshape(0, dimensionality),
        "nr": 0,
        "label": "root"
    }

    ##################################################
    # PREPARE FOR PARENT/CHILD SEARCH
    ##################################################
    for i, cluster in enumerate(final_clusters):

        # Assign stats to clusters (lambda, center, nr)
        # ---------------------------------
        cluster["lambda"] = dimensionality - cluster["w_c"].sum()
        cluster["center"] = cluster["data"].mean(axis=0)
        cluster["nr"] = len(cluster["data"])
        cluster["child_of"] = []

        # Combine small clusters to noise
        # ---------------------------------
        if cluster["nr"] < mu:
            print("noise found")
            noise["data"] = np.vstack( (noise["data"], cluster["data"]) )
            noise["center"] = noise["data"].mean(axis=0)
            noise["nr"] += cluster["nr"]

    # remove noise clusters
    final_clusters[:] = [val for val in final_clusters if val["nr"] > mu]

    # Sort according to lambda
    final_clusters[:] = sorted(final_clusters, key=lambda k: k['lambda'])

    # label the cluster according to w(c)
    # ---------------------------------
    for cluster in final_clusters:
        proposed_label = str(cluster["w_c"].astype(int).tolist())
        cluster["label"] = proposed_label

    # get rid of duplicated labels
    for k, cluster in enumerate(final_clusters):
        i = 1
        LABLE_CLASH = False

        for cluster_j in final_clusters[k+1:]:
            if cluster["label"] == cluster_j["label"]:
                print("label clash")
                LABLE_CLASH = True
                cluster_j["label"] += "_" + str(i)
                i += 1

        if LABLE_CLASH:
            cluster["label"] += "_" + str(i)

    #

    ##################################################
    # Find parents and childs
    ##################################################
    # for all clusters, starting with highest dimensionality
    for i, cluster in enumerate(reversed(final_clusters)):
        print("main", cluster["label"], cluster["lambda"])
        max_lambda = dimensionality

        # for all cluster with higher dimensionality then that
        for higher_cluster in final_clusters[-(i-1):]:
            # print(" - sub", higher_cluster["label"], higher_cluster["lambda"] > cluster["lambda"])
            if higher_cluster["lambda"] > cluster["lambda"]:
                # print(" - sub", higher_cluster["label"])

                w_cc = cluster["w_c"] * higher_cluster["w_c"]
                dist = DIST_projected(cluster["center"], higher_cluster["center"], preference_matrix=w_cc)

                # only assign parent if near enough
                # only assign parent if there is no cluster lower in hirachy that is the parent
                if (higher_cluster["lambda"] <= max_lambda) and (dist < 2*epsilon):
                    print("parent found: "+str(higher_cluster["label"]))
                    cluster["child_of"] += [higher_cluster["label"]]
                    max_lambda = higher_cluster["lambda"]

        if len(cluster["child_of"]) == 0:
            print("parent found: noise")
            cluster["child_of"] += [noise["label"]]

    #

    ##################################################
    # Plotting
    ##################################################
    if PLOT_RESULTS:
        try:
            import networkx as nx
            G = nx.DiGraph()
            for cluster in final_clusters:
                for child in cluster["child_of"]:
                    G.add_edge(cluster["label"], child)

            fig, ax = plt.subplots()
            pos = nx.kamada_kawai_layout(G)
            nx.draw(G, pos, alpha=1, with_labels=True, font_size=8, ax=ax)
            ax.set_title("Graph display of hierarchy")
            plt.show()

            # fig, ax = plt.subplots(2)
            # pos = nx.kamada_kawai_layout(G)
            # nx.draw(G, pos, alpha=1, with_labels=True, font_size=8, ax=ax[0])
            # ax[0].set_title("Graph display of hierarchy - Alternative 1")
            # plt.show()

            # # Experiment: position of nodes
            # # ----------------------------------------------------------
            # pos = nx.spectral_layout(G)
            #
            # for node in pos:
            #     cluster = [cl for cl in final_clusters if cl["label"].replace(",", "") == node.replace(",", "")]
            #
            #     if cluster == []:
            #         pos[node] = np.array([0, 1.5])
            #     else:
            #         try:
            #             pos[node][1] = cluster[0]["lambda"]
            #         except:
            #             pos[node] = np.hstack((pos[node], cluster[0]["lambda"]))
            # nx.draw(G, pos, alpha=1, with_labels=True, font_size=8, ax=ax[1])
            # ax[1].set_title("Graph display of hierarchy - Alternative2 \n(overlapping might occur)")
            # plt.show()
        except:
            print("Some error occured at the creation of the clusters. Please ensure that networkx is installed")
        # ----------------------------------------------------------

    return final_clusters


#


# ------------------------
#  PLOTTING
# ------------------------


def plot_preference_vectors(data, preference_vector):
    pref_ax0 = data[preference_vector[:, 0]]
    pref_ax1 = data[preference_vector[:, 1]]

    fig, ax = plt.subplots()
    fig.suptitle("Preference Vector Plot (2D)")
    ax.plot(data[:, 0], data[:, 1], "ko", markersize=10)
    ax.plot(pref_ax0[:, 0], pref_ax0[:, 1], "r^", label="w_up == True", markersize=10)
    ax.plot(pref_ax1[:, 0], pref_ax1[:, 1], "y>", label="w_right == True", markersize=10)
    return fig, ax


def plot_reachablity_plot(pq):
    fig, ax = plt.subplots()
    fig.suptitle("Reachability Plot")
    ax.plot(pq[:, 1], '.', color="black", label="RDist")
    ax.fill_between(np.arange(0, pq.shape[0]), pq[:, 1], 0, color="black")
    ax.legend(loc="upper center")
    return fig, ax


def plot_cluster(data, cluster_list, mu):

    fig, ax = plt.subplots()
    fig.suptitle("DiSH Clustering Results")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ax.plot(data[:, 0], data[:, 1], 'ko')

    for i, cluster in enumerate(cluster_list):
        cluster_data = cluster["data"]
        cluster_center = cluster["data"].mean(axis=0)

        if cluster_data.shape[0] < mu:
            ax.plot(cluster_data[:, 0], cluster_data[:, 1], 'ko')
        else:
            ax.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', color=colors[i % 9],
                    label="Cluster " + str(cluster["w_c"]))
            ax.plot(cluster_center[0], cluster_center[1], 'x', markersize=20, color=colors[i % 9])

    ax.legend(loc="upper right")
    return fig, ax


def plot_sdist(point, data, sdist_p, fig, ax):
    cmaps = ["Reds_r", "Blues_r", "Greens_r"]
    for i in range(3):
        indices = np.argwhere(sdist_p[0] == i)
        print(i, len(indices))
        ax.scatter(x=data[indices, 0], y=data[indices, 1], c=sdist_p[1][indices], s=20, cmap=cmaps[i])

    ax.plot(point[0], point[1], 'ro', label="p")
    fig.suptitle("COLOR - MAP\nd1=0 (red),   d1=1 (blue),   d1=2 (green)\n Darker color means lower d2")
    ax.plot()
    return fig, ax

#


if __name__ == '__main__':

    # import data
    # ------------------
    sep = " "
    fpath = r"../datasets/mouse.csv"

    dataframe = pd.read_csv(fpath, sep=sep, comment="#", header=None)
    dataframe = dataframe.dropna(axis=1, how='all')
    data = dataframe.values

    # Run DiSH
    # ------------------
    mu = 25
    epsilon = 0.13
    cluster_list = dish(data, epsilon=epsilon, mu=mu)
    final_cluster = build_hirarchy(cluster_list, mu=mu, epsilon=epsilon)
    # plot_cluster(data, cluster_list, mu=mu)


    # Evaluation
    # ------------------

    def evaluate_clusters(data, cluster_list, mu):
        from sklearn import metrics
        data_clust1 = data[:290]
        data_clust2 = data[290:390]
        data_clust3 = data[390:490]
        data_noise = data[490:500]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data_clust1[:, 0], data_clust1[:, 1], '.', color='red')
        ax[0].plot(data_clust2[:, 0], data_clust2[:, 1], '.', color='blue')
        ax[0].plot(data_clust3[:, 0], data_clust3[:, 1], '.', color='green')
        ax[0].plot(data_noise[:, 0], data_noise[:, 1], '.', color='black')
        ax[0].set_title("True Labels")

        ax[1].set_title("Clustering Results")
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = ['red', 'blue', 'green']  # prop_cycle.by_key()['color']
        for i, cluster in enumerate(cluster_list):
            cluster_data = cluster["data"]
            cluster_center = cluster["data"].mean(axis=0)

            if cluster_data.shape[0] < mu:
                ax[1].plot(cluster_data[:, 0], cluster_data[:, 1], 'k.')

            else:
                ax[1].plot(cluster_data[:, 0], cluster_data[:, 1], '.', color=colors[i % 9],
                           label="Cluster " + str(cluster["w_c"]))
                ax[1].plot(cluster_center[0], cluster_center[1], 'x', markersize=20, color=colors[i % 9])
        plt.savefig("cluster_results.png")

        labels_true = [1 for i in range(0, 290)]\
                      + [2 for i in range(290, 390)]\
                      + [3 for i in range(390, 490)]\
                      + [0 for i in range(490, 500)]

        labels_pred = []
        for point in data:
            pointfound = False
            for cluster in cluster_list:
                for point2 in cluster["data"]:
                    if np.isclose(point[0], point2[0]) and np.isclose(point[1], point2[1]):
                        if cluster["data"].shape[0] < mu:
                            labels_pred.append(0)  # noise label
                        else:
                            labels_pred.append(cluster["num"])  # cluster label

                        pointfound = True
            if pointfound == False:
                print("point not found...")

        scores = metrics.mutual_info_score(labels_true, labels_pred), metrics.adjusted_mutual_info_score(labels_true,
                                                                                                         labels_pred)
        print("scores:", scores)
        return scores

    evaluate_clusters(data=data, cluster_list=cluster_list, mu=mu)