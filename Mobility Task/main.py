"""
"""


import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Attention: use conda to install (I use Pycharm with a Conda Interpreter) Version 0.17.1
# SOURCE: https://scitools.org.uk/cartopy/docs/latest/installing.html
import cartopy.crs as ccrs
import cartopy.io.img_tiles as map_img


def main():

    # PARAMETER
    # --------------------------------------------
    data_directory = r"./cgt_stud_2019"               # local path
    subdir = r"10_83_2019-10-31T084235.379"
    dir = os.path.join(data_directory, subdir)
    # --------------------------------------------

    # Read Files
    markers = read_markers_csv(dir)
    acceleration = read_acceleration_csv(dir)
    positions = read_position_csv(dir)

    # Calculations
    mode_changes = get_mode_changes(markers)

    # Plotting
    plot_acceleration(acceleration, mode_changes=mode_changes, title=dir)
    plot_position(positions)
    # plot_acceleration_and_gps(acceleration, positions)
    return True


#%%


def read_position_csv(dir):
    fname = os.path.join(dir, "positions.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    return data


def read_acceleration_csv(dir):
    fname = os.path.join(dir, "acceleration.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    return data


def read_markers_csv(dir):
    fname = os.path.join(dir, "markers.csv")
    columns = ["time", "key", "value", "value1", "value2", "value3", "value4", "value5", "value6", "value7", "value8"]
    data = pd.read_csv(fname, sep=";", engine="python", names=columns, parse_dates=["time"])
    return data


def get_mode_changes(markers):
    mode_changed = markers[markers["key"] == "CGT_MODE_CHANGED"]
    mode_changed = mode_changed.iloc[:,:9]
    columns = ["time", "key", "datetime", "mode", "longitude", "latitude", "unknown", "station_name", "unknown2"]
    mode_changed.columns = columns
    return mode_changed


#%


def plot_acceleration(acceleration, ax=None, mode_changes=None, title=""):

    if ax is None:
        fig, ax = plt.subplots()

    max_y = max(acceleration["x"].max(), acceleration["y"].max(), acceleration["z"].max())
    ax.plot(acceleration["time"], acceleration["x"])
    ax.plot(acceleration["time"], acceleration["y"])
    ax.plot(acceleration["time"], acceleration["z"])

    major_ticks = mdates.MinuteLocator(byminute=range(0, 60, 5))
    major_formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(major_ticks)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.set_ylabel(u"Acceleration \n [m/s²]")
    ax.set_title("Acceleration data \n" + title)

    if mode_changes is not None:
        for i, mode_change in mode_changes.iterrows():
            ax.axvline(x=mode_change["time"], ymin=-100, ymax=100, color="k", linestyle="--")
            ax.text(s=" "+mode_change["mode"], x=mode_change["time"], y=max_y*0.8)
    return ax


def plot_position(positions, fig=None, subplot_nrRows=1, subplot_nrCols=1, subplot_index=1):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot_nrRows, subplot_nrCols, subplot_index, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(positions["longitude"], positions["latitude"], color="green")

    ax.add_image(map_img.GoogleTiles(),         # Service
                 1,                            # Zoomlevel
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
