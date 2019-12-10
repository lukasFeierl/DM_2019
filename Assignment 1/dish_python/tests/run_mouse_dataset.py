import pandas as pd

# import data
# ------------------
sep = " "
fpath = r"../../datasets/mouse.csv"

dataframe = pd.read_csv(fpath, sep=sep, comment="#", header=None)
dataframe = dataframe.dropna(axis=1, how='all')
data = dataframe.values



# Run DiSH
# ------------------
import sys
sys.path.append("..")
from dish import dish, build_hirarchy

mu = 25
epsilon = 0.13
cluster_list = dish(data, epsilon=epsilon, mu=mu)
final_cluster = build_hirarchy(cluster_list, mu=mu, epsilon=epsilon)
plot_cluster(data, cluster_list, mu=mu)