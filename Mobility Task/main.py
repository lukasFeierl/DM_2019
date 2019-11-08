"""
"""


import os
import pandas as pd


def main():
    # PARAMETER
    # --------------------------------------------
    data_directory = r"./cgt_stud_2019"               # local path
    subdir = r"10_83_2019-10-31T084235.379"
    dir = os.path.join(data_directory, subdir)
    # --------------------------------------------

    acceleration = read_acceleration_csv(dir)
    positions = read_position_csv(dir)
    plot_acceleration(acceleration)
    # plot_acceleration_and_gps(acceleration, positions)
    return True


#%%


def read_position_csv(dir):
    fname = os.path.join(dir, "positions.csv")
    data = pd.read_csv(fname)
    return data


def read_acceleration_csv(dir):
    fname = os.path.join(dir, "acceleration.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    return data



import matplotlib.dates as mdates
def plot_acceleration(acceleration, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(acceleration["time"], acceleration["x"])
    ax.plot(acceleration["time"], acceleration["y"])
    ax.plot(acceleration["time"], acceleration["z"])

    major_ticks = mdates.MinuteLocator(byminute=range(0, 60, 5))
    major_formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(major_ticks)
    ax.xaxis.set_major_formatter(major_formatter)
    return ax


# Attention: use conda to install (I use Pycharm with a Conda Interpreter) Version 0.15.1
# SOURCE: https://scitools.org.uk/cartopy/docs/latest/installing.html
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as map_img


def plot_position(positions, fig=None, subplot_nrRows=1, subplot_nrCols=1, subplot_index=1):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot_nrRows, subplot_nrCols, subplot_index, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(positions["longitude"], positions["latitude"], color="green")
    ax.add_image(map_img.GoogleTiles(),         # Service
                 14,                            # Zoomlevel
                 # interpolation='spline36',      # Interpolation (more accurate)
                 # regrid_shape=2000              # making interpolation smoother
                 )                              # funktioniert nur mit internetverbindung
    # ax.add_wms(
    #            # wms='http://labs.metacarta.com/wms/vmap0',
    #            wms='http://vmap0.tiles.osgeo.org/wms/vmap0',
    #            layers=['basic', 'secroad', 'priroad', "rail"])
    return ax



def plot_acceleration_and_gps(acceleration, positions):
    fig = plt.figure(figsize=(10, 5))
    ax = plot_position(positions, fig, subplot_nrRows=1, subplot_nrCols=2)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_acceleration(acceleration, ax)

#%%


if __name__ == '__main__':
    main()
