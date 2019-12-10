# -------------------------------------------------------------------
# Import DiSH
# -------------------------------------------------------------------
import sys
sys.path.append("..")
from dish_main import dish, build_hirarchy           # This might be displayed as an error but should work at runtime.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------
# select the dataset you want to apply DiSH on

#
sep = ","
fpath = r"../../datasets/testset_b.txt"
mu = 5           # 50
epsilon = 0.5      # 0.05


sep = ";"
fpath = r"../../datasets/lines_and_noise.csv"
# best looking result: mu=90; eps=0.1
# best logical result: mu= 2; eps=0.005
mu = 90
epsilon = 0.1
mu = 2              # 50
epsilon = 0.005      # 0.05


sep = " "
fpath = r"../../datasets/simple_lines.csv"
mu = 3
epsilon = 0.1


sep = " "
fpath = r"../../datasets/mouse.csv"
mu = 25                     # 40
epsilon = 0.13              # with 0.12 you just look for 0.12 wide circles in the data :(
# -------------------------------------------------------------------


#%% -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

# import data
dataframe = pd.read_csv(fpath, sep=sep, comment="#", header=None)
dataframe = dataframe.dropna(axis=1,how='all')
data = dataframe.values



#%%
# Test-plot DIST
# ----------------
from dish_main import DIST
fig, ax = plt.subplots()
p = data[0]
q = data[1]
fig.suptitle("Distance between points - DIST")
ax.plot(data[:, 0], data[:, 1], 'k.')
ax.plot(p[0], p[1], 'ro', label="p")
ax.plot(q[0], q[1], 'rx', label="q")
ax.plot([p[0], q[0]], [p[1], q[1]], label="DIST(p,q)"+str(DIST(p, q[np.newaxis])))
ax.plot([p[0], p[0]], [p[1], p[1]], label="DIST(p,p)"+str(DIST(p, p[np.newaxis])))
ax.legend()
# ----------------



#%%
# Test-plot Neighbors
# ----------------
from dish_main import get_neighbors
p = data[0]
fig, ax = plt.subplots()
fig.suptitle("Neighbors of point p - get_neighbors")
ax.plot(data[:, 0], data[:, 1], 'k.')
ax.plot(p[0], p[1], 'ro', label="Point p")
for features in [[0], [1], [0,1]]:
    neighbors = get_neighbors(p, data, features=features, epsilon=epsilon)
    ax.plot(neighbors[:, 0], neighbors[:, 1], '.', label="feature-dimension: "+str(features))

plt.legend()
# ----------------


#%%
# Test-plot Subspace
# --------------------
from dish_main import get_best_subspace
point = data[0]
get_best_subspace(point, data, mu, epsilon, TEST_PLOT=True)
# --------------------


#%%
# Test Preference vectors
# --------------------
from dish_main import plot_preference_vectors, get_preference_vectors
preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)
plot_preference_vectors(data, preference_vector)
# --------------------



# Test-plot DIST_projected
# ----------------
from dish_main import DIST_projected
fig, ax = plt.subplots()
p = data[0]
q = data[1]

ax.plot(data[:, 0], data[:, 1], 'k.')
ax.plot(p[0], p[1], 'ro', label="p")
ax.plot(q[0], q[1], 'rx', label="q")
ax.plot([p[0], q[0]], [p[1], p[1]], label="DIST_[0] (p,q)"+str(DIST_projected(p, q[np.newaxis], preference_matrix=[True, False])))
ax.plot([p[0], p[0]], [p[1], q[1]], label="DIST_[1] (p,p)"+str(DIST_projected(p, q[np.newaxis], preference_matrix=[False, True])))
ax.plot([p[0], q[0]], [p[1], q[1]], label="DIST_[0,1] (p,p)"+str(DIST_projected(p, q[np.newaxis], preference_matrix=[True, True])))
ax.legend()
# ----------------




# Plot Test : Subspace Distance
# ------------------------------
from dish_main import get_subspace_distance, get_preference_vectors, plot_sdist
p_index = 0

point = data[p_index]
preference_vector = get_preference_vectors(data, mu, epsilon)
sdist_p = get_subspace_distance(data, p_index, preference_vector, epsilon)
fig, ax = plt.subplots()
plot_sdist(point, data, sdist_p, fig, ax)
# ------------------------------




# Plot Test : Reachablity Distance
# ------------------------------
from dish_main import get_subspace_distance, get_reachability_distance, plot_sdist
p_index = 0

point = data[p_index]
preference_vector = get_preference_vectors(data, mu, epsilon)
d1, d2 = get_subspace_distance(data, p_index, preference_vector, epsilon)
sdist_p = get_reachability_distance(d1, d2, mu=mu)

fig, ax = plt.subplots()
plot_sdist(point, data, sdist_p, fig, ax)
# ------------------------------




# Test : pq update
# ------------------------------
from dish_main import update_pq
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


# PLot reachability plot
# ------------------------------
from dish_main import get_preference_vectors, get_pq, plot_reachablity_plot
preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)
pq = get_pq(data, preference_vector, epsilon=epsilon, mu=mu)
plot_reachablity_plot(pq)
# ------------------------------


# Run Main Algorithm
# ------------------------------
from dish_main import dish, build_hirarchy, plot_cluster
cluster_list = dish(data, epsilon=epsilon, mu=mu)
final_cluster = build_hirarchy(cluster_list, mu=mu, epsilon=epsilon)
fig, ax = plot_cluster(data, cluster_list, mu=mu)
# ------------------------------


# No assignment to noise
# ------------------------------
from dish_main import dish, build_hirarchy, plot_cluster
cluster_list = dish(data, epsilon=epsilon, mu=mu)
final_cluster = build_hirarchy(cluster_list, mu=mu, epsilon=epsilon)
fig, ax = plot_cluster(data, cluster_list, mu=1)
# ------------------------------



# Plot cluster order:
# ----------------------------
from dish_main import dish, get_pq, plot_cluster, get_preference_vectors
cluster_list = dish(data, epsilon=epsilon, mu=mu)
preference_vector = get_preference_vectors(data, mu=mu, epsilon=epsilon)
pq = get_pq(data, preference_vector, epsilon=epsilon, mu=mu)

for i, index in enumerate(pq[:, 0]):
    index = int(index)
    ax.text(x=data[index][0], y=data[index][1] + 0.08, s=str(i))


# Used for testplot
fig, ax = plot_cluster(data, cluster_list, mu=1)
ax.legend().remove()
p = int(pq[49, 0])
q = int(pq[50, 0])
z = int(pq[51, 0])

preference_vector[p]
l1, = ax.plot(data[p, 0], data[p, 1], 'X', label="p", color="red")
l2, = ax.plot(data[q, 0], data[q, 1], 'X', label="q", color="#e26900")
l3, = ax.plot(data[z, 0], data[z, 1], 'X', label="z", color="#6088ff")
ax.legend([l1, l2, l3], ["p", "q", "z"])
# ----------------------------
