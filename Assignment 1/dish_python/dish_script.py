import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------
# Problems:
# - (lines_and_nois.csv)    TODO: not able to connect lines... At least if they are not axis-parallel.
# - Hard to find clusters that are not parallel to axes (lines_and_noise.csv)
# - hard to detect varying densities
# - how to set mu and epsilon (when not knowing the result) ?!!
# - mu-nearest neighbor not working in our code?
# - maybe the problem is noise??
# - maybe more useful in more dimension, where overlaying does not occur so often ??
#
# -------------------------------------------------------------------


# sep = ";"
# fpath = r"../datasets/lines_and_noise.csv"
# # best looking result: mu=90; eps=0.1
# # best logical result: mu= 2; eps=0.005
# mu = 2              # 50
# epsilon = 0.005      # 0.05


# sep = " "
# fpath = r"../datasets/simple_lines.csv"
# mu = 3
# epsilon = 0.1


sep = " "
fpath = r"../datasets/mouse.csv"
mu = 40                     # 40
epsilon = 0.1              # with 0.12 you just look for 0.12 wide circles in the data :(

# -------------------------------------------------------------------


#


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

# import data
dataframe = pd.read_csv(fpath, sep=sep, comment="#", header=None)
dataframe = dataframe.dropna(axis=1,how='all')
data = dataframe.values


def DIST(point, data):
    """ calculates the euclidean distance between point and all points in data"""
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


# # Test-plot DIST
# # ----------------
# fig, ax = plt.subplots()
# p = data[0]
# q = data[1]
# ax.plot(data[:, 0], data[:, 1], 'k.')
# ax.plot(p[0], p[1], 'ro', label="p")
# ax.plot(q[0], q[1], 'rx', label="q")
# ax.plot([p[0], q[0]], [p[1], q[1]], label="DIST(p,q)"+str(DIST(p, q[np.newaxis])))
# ax.plot([p[0], p[0]], [p[1], p[1]], label="DIST(p,p)"+str(DIST(p, p[np.newaxis])))
# ax.legend()
# # ----------------


def get_neighbors(point, data, features, epsilon):
    """ returns all neighbors of a point, if projected along features in a radius epsilon"""
    is_near = DIST(data[:, features], point[features]) <= epsilon         # point[features] = projection to one feature
    return data[is_near]

# # Test-plot Neighbors
# # ----------------
# p = data[0]
# fig, ax = plt.subplots()
# fig.suptitle("Neighbors of point p")
# ax.plot(data[:, 0], data[:, 1], 'k.')
# ax.plot(p[0], p[1], 'ro', label="Point p")
# for features in [[0], [1], [0,1]]:
#     neighbors = get_neighbors(p, data, features=features, epsilon=epsilon)
#     ax.plot(neighbors[:, 0], neighbors[:, 1], '.', label="feature-dimension: "+str(features))
#
# plt.legend()
# # ----------------


def get_best_subspace(point, data, mu, epsilon, TEST_PLOT=False):
    """ calculates the best subspace for a given point. An attribute/feature contributes to the best subspace if enough
    elements are in the neighborhood of the point in a radius epsilon. """

    #-----------------------------
    if TEST_PLOT:
        fig, ax = plt.subplots()
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
            ax.plot(neighbors[:,0], neighbors[:,1], ".", label="feature: "+ str(feature)+" (#"+str(len(neighbors))+")")
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
        proposed_combination = best_subspace + [best_feature]   # add feature and subspace to form new proposed-subspace

        # Check if criterion holds true for proposed subspace
        neighbors = get_neighbors(point, data=data, features=proposed_combination, epsilon=epsilon)
        if len(neighbors) >= mu:
            best_subspace = proposed_combination

    # -----------------------------
    if TEST_PLOT:
        neighbors = get_neighbors(point, data=data, features=best_subspace, epsilon=epsilon)
        ax.plot(neighbors[:, 0], neighbors[:, 1], '.', label="best subspace: "+ str(best_subspace)+" (#"+str(len(neighbors))+")")
        ax.legend()
    # -----------------------------

    return best_subspace


# # Test-plot Subspace
# # --------------------
# point = data[0]
# get_best_subspace(point, data, mu, epsilon, TEST_PLOT=True)
# # --------------------


def get_preference_vectors(data, mu, epsilon):
    """ calculates the preference vector w(p) of each point, assigning "True" to each feature which is in the "best
    subspace" - thus relevant for clustering - and "False" otherwise.
    The number of False-values in the preference vector is the called the subspace dimensionality Lambda.
    """
    # initialising - stores w(p) for each point
    preference_vector = np.zeros(data.shape, dtype=bool)

    for point_index, point in enumerate(data):
        # calculate which features are relevant
        is_best_subspace = get_best_subspace(point, data, mu=mu, epsilon=epsilon)
        preference_vector[point_index][is_best_subspace] = True

    return preference_vector


# Test-plot Preference Vectors
# --------------------
# done late with plot_preference_vectors()
# --------------------


def DIST_projected(point, data, preference_matrix):
    """ calculates the euclidean distance, but uses the preference_vectors to project the data to lower dimension.
    Returns distances as a vector for each distance p-q for each point q in the data"""
    dist_matrix = (point - data) ** 2

    # Project to feature dimensions specified in preference Matrix
    projected_dist_matrix = np.multiply(dist_matrix, preference_matrix)

    if len(data.shape) == 1:
        dist_vector = np.sqrt(np.sum(projected_dist_matrix))
    else:
        dist_vector = np.sqrt(np.sum(projected_dist_matrix, axis=1))

    return dist_vector


# # Test-plot DIST_projected
# # ----------------
# fig, ax = plt.subplots()
# p = data[0]
# q = data[1]
#
# ax.plot(data[:, 0], data[:, 1], 'k.')
# ax.plot(p[0], p[1], 'ro', label="p")
# ax.plot(q[0], q[1], 'rx', label="q")
# ax.plot([p[0], q[0]], [p[1], p[1]], label="DIST_[0] (p,q)"+str(DIST_projected(p, q[np.newaxis], preference_matrix=[True, False])))
# ax.plot([p[0], p[0]], [p[1], q[1]], label="DIST_[1] (p,p)"+str(DIST_projected(p, q[np.newaxis], preference_matrix=[False, True])))
# ax.plot([p[0], q[0]], [p[1], q[1]], label="DIST_[0,1] (p,p)"+str(DIST_projected(p, q[np.newaxis], preference_matrix=[True, True])))
# ax.legend()
# # ----------------


def get_subspace_distance(data, p_index, preference_vector, epsilon):
    """ calculates the subspace distance between the point and all the other points in the data.
    The subspace distance consists of two values d1 and d2. d1 represents the dimensionality in respect to another
    point. Two points have the same d1 if they are k.dim clusters belonging to the same cluster, or k-1 dim. clusters
    belonging to separate clusters (i.e not the same preference vector or to far away from each other).
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


# # Plot Test : Subspace Distance
# # ------------------------------
# p_index = 0
#
# point = data[p_index]
# preference_vector = get_preference_vectors(data, mu, epsilon)
# sdist_p = get_subspace_distance(data, p_index, preference_vector, epsilon)
#
# fig, ax = plt.subplots()
# cmaps = ["Reds_r", "Blues_r", "Greens_r"]
# for i in range(3):
#     indices = np.argwhere(sdist_p[0] == i)
#     print(i, len(indices))
#     ax.scatter(x=data[indices, 0], y=data[indices, 1], c=sdist_p[1][indices], s=20, cmap=cmaps[i])
#
# ax.plot(point[0], point[1], 'ro', label="p")
# fig.suptitle("COLOR - MAP\nd1=0 (red),   d1=1 (blue),   d1=2 (green)\n Darker colour means lower d2")
# ax.plot()
# # ------------------------------


def get_reachability_distance(d1, d2, mu):
    """ to avoid single link effect, the sdist of the mu nearest neighbor of the point in respect to p is used
    as minimum sdist. If the point is in a cluster with less then mu neighbors this then results in beeing a
    one-point cluster (hence, no cluster) """

    d = np.vstack((d1, d2)).T
    d_argsort = np.lexsort((d[:, 1], d[:, 0]))  # Sort according to d1, then d2
    d_sorted = d[d_argsort]
    d_mu = d_sorted[mu]
    d_sorted[:mu] = d_mu       # is equal to max(sdist(p, r), sdist(p, mu))
    d[d_argsort] = d_sorted

    return d[:, 0], d[:, 1]


# Plot Test : Reachablity Distance
# ------------------------------
def plot_sdist(point, data, sdist_p, fig, ax):
    cmaps = ["Reds_r", "Blues_r", "Greens_r"]
    for i in range(3):
        indices = np.argwhere(sdist_p[0] == i)
        print(i, len(indices))
        ax.scatter(x=data[indices, 0], y=data[indices, 1], c=sdist_p[1][indices], s=20, cmap=cmaps[i])

    ax.plot(point[0], point[1], 'ro', label="p")
    fig.suptitle("COLOR - MAP\nd1=0 (red),   d1=1 (blue),   d1=2 (green)\n Darker colour means lower d2")
    ax.plot()

# p_index = 0
#
# point = data[p_index]
# preference_vector = get_preference_vectors(data, mu, epsilon)
# d1, d2 = get_subspace_distance(data, p_index, preference_vector, epsilon)
# sdist_p = get_reachability_distance(d1, d2, mu=mu)
#
# fig, ax = plt.subplots()
# plot_sdist(point, data, sdist_p, fig, ax)
# ------------------------------


def update_pq(pq, d1, d2):
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

    # # Old version which is wrong... but yields better results...
    # is_minimum = np.zeros(d1_old_new.shape, dtype=bool)  # initialising array
    # minimum_value = np.nanmin(d1_old_new, axis=0)
    # is_minimum[d1_old_new == minimum_value] = True
    # is_minimum[:, d1_is_equal] = False
    #
    # pq[:, 1][~d1_is_equal] = d1_old_new[is_minimum]
    # pq[:, 2][~d1_is_equal] = d2_old_new[is_minimum]

    return pq


# Test : pq update
# ------------------------------
pq = np.ones((6, 3))
pq[:, 0] = np.arange(0, pq.shape[0])
d1 = np.array([0, 1, 2,   1,  1, np.NaN])
d2 = np.array([0, 0, 0, 1.1, 0.9, np.NaN])

pq_new = update_pq(pq=pq, d1=d1, d2=d2)

pq_result = np.array([[0., 0., 0.],
                      [1., 1., 0.],
                      [2., 1., 1.],
                      [3., 1., 1.],
                      [4., 1., 0.9],
                      [5., 1., 1.],
                      ])

assert((pq_new == pq_result).all())
# ------------------------------


def get_pq(data, preference_vector, epsilon, mu, PLOT_TEST=False):

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


# Test : pq sort
# ------------------------------
pq = np.array([[0., 0., 0.],
               [1., 1., 0.],
               [2., 1., 1.],
               [3., 0., 1.],
               [4., 1., 0.9],
                ])
pq_result = np.array([ [0., 0., 0.],
                       [3., 0., 1.],
                       [1., 1., 0.],
                       [4., 1., 0.9],
                       [2., 1., 1.],
                       ])
shift_index = 0
index = 0
pq_sort = pq[index + shift_index:]  # only sort not yet visited points !!
pq_argsort = np.lexsort((pq_sort[:, 2], pq_sort[:, 1]))  # Sort according to d1, then d2
pq[index + shift_index:] = pq_sort[pq_argsort]  # Apply reordering
assert((pq == pq_result).all())
# ------------------------------

# PLOT Test PQ
# ------------------------------
preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)
get_pq(data, preference_vector, epsilon, mu, PLOT_TEST=False)

preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)
pq = get_pq(data, preference_vector, epsilon=epsilon, mu=mu)

# ------------------------------


def dish(data, epsilon, mu):
    """ Starts the DiSH Algorithm"""

    # Calculate Preference Vectors
    # ------------------
    preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)  # w(p) as array for each row (i.e. point) of the data

    # Calculate ReachDistances
    # ------------------
    pq = get_pq(data, preference_vector, epsilon=epsilon, mu=mu)

    # Extract Cluster
    # ------------------
    cluster_order = pq  # cols: [point_index, d1, d2]
    cluster_list = []  # list with individual clusters (containing numpy arrays with all points of the cluster)

    predecessor = cluster_order[0]  # first point / previous point

    for object in cluster_order:
        o_index = int(object[0])  # object is a point with [index, d1 and d2]
        p_index = int(predecessor[0])

        point_o = data[o_index]

        w_o = preference_vector[o_index]
        w_p = preference_vector[p_index]
        w_op = w_p * w_o

        # Get corresponding cluster
        # ---------------------------
        corresponding_cluster = None
        for cluster in cluster_list:
            c_center = cluster["data"].mean(axis=0)

            has_same_preference_vector = (cluster["w_c"] == w_op).all()
            is_near_enough = DIST_projected(point_o, c_center, preference_matrix=w_op) <= 2*epsilon

            if has_same_preference_vector and is_near_enough:
                corresponding_cluster = cluster
                cluster["data"] = np.vstack((cluster["data"], point_o))
                break

        if corresponding_cluster is None:
            print("Cluster "+str(len(cluster_list)+1)+" found.")
            cluster_data = np.array(point_o)[np.newaxis]
            cluster_w_c = w_o
            cluster_list += [{"data": cluster_data,
                              "w_c": cluster_w_c}]
        predecessor = object

    return cluster_list


preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)
pq = get_pq(data, preference_vector, epsilon=epsilon, mu=mu)
cluster_list = dish(data, epsilon=epsilon, mu=mu)


#


# ------------------------
#  PLOTTING
# ------------------------


def plot_reference_vectors(data, preference_vector):
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


def plot_cluster(cluster_list, mu):
    fig, ax = plt.subplots()
    fig.suptitle("DiSH Clustering Results")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

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

# plot_reference_vectors(data, preference_vector)
# plot_reachablity_plot(pq)
plot_cluster(cluster_list, mu=mu)