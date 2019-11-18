import pandas as pd


#%%
# DATASETS
# -------------------------------------------------------------------

# 3 Lines
fpath = r"../datasets/simple_lines.csv"
mu = 3
epsilon = 0.1

# Elki Mouse dataset
fpath = r"../datasets/mouse.csv"
mu = 40
epsilon = 0.1

# -------------------------------------------------------------------


#%%
# GET DATA
# -------------------------------------------------------------------
dataframe = pd.read_csv(fpath, sep=" ", comment="#", header=None)
data = dataframe.values
# -------------------------------------------------------------------


#%%
# Run DiSH
# -------------------------------------------------------------------
from dish_class import DiSH
self = DiSH(mu=mu, epsilon=epsilon)
cl_list = self.fit(data=data, SHOW_PLOT=True)
# -------------------------------------------------------------------


