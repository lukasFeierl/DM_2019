"""
"""


import os
import pandas as pd

# PARAMETER

data_directory = r"./cgt_stud_2019"         # local path
subdir = r"10_83_2019-10-31T084235.379"
dir = os.path.join(data_directory, subdir)


#%%


def read_position_csv(dir):
    fname = os.path.join(dir, "positions.csv")
    data = pd.read_csv(fname)
    return data


def read_acceleration_csv(dir):
    fname = os.path.join(dir, "acceleration.csv")
    data = pd.read_csv(fname)
    return data


positions = read_position_csv(dir)
acceleration = read_acceleration_csv(dir)


#%%


acceleration.plot()

# Attention: use conda to install (I use Pycharm with a Conda Interpreter) Version 0.15.1
# SOURCE: https://scitools.org.uk/cartopy/docs/latest/installing.html
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as map_img


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()

# Set extent to Austria
ax.scatter(positions["longitude"], positions["latitude"], color="green")
ax.add_image(map_img.GoogleTiles(), 15)             # funktioniert nur mit internetverbindung


# ax.add_wms(
#            # wms='http://labs.metacarta.com/wms/vmap0',
#            wms='http://vmap0.tiles.osgeo.org/wms/vmap0',
#            layers=['basic', 'secroad', 'priroad', "rail"])
