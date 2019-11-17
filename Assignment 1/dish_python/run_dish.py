
from dish import dish


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
dish(data=data, mu=mu, epsilon=epsilon)
# -------------------------------------------------------------------


