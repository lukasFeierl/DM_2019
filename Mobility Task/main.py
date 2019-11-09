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
    subdir_list = [r"10_78_2019-11-01T180450.475",
                   r"10_82_2019-10-31T085843.034",
                   r"10_83_2019-10-31T084235.379",
                   ]
    # --------------------------------------------

    for subdir in subdir_list:
        print(subdir)
        dir = os.path.join(data_directory, subdir)

        # Read Files
        markers = _read_markers_csv(dir)
        acceleration = _read_acceleration_csv(dir)
        positions = _read_position_csv(dir)

        # Calculations
        mode_changes = _get_mode_changes(markers)

        # Plotting
        plot_acceleration(acceleration, mode_changes=mode_changes, title=dir)
        plot_position(positions)
        # plot_acceleration_and_gps(acceleration, positions, mode_changes=mode_changes, title=dir)
    return True


#%%


def _read_position_csv(dir):
    fname = os.path.join(dir, "positions.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    return data


def _read_acceleration_csv(dir):
    fname = os.path.join(dir, "acceleration.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    return data


def _read_markers_csv(dir):
    fname = os.path.join(dir, "markers.csv")
    columns = ["time", "key", "value", "value1", "value2", "value3", "value4", "value5", "value6", "value7", "value8"]
    data = pd.read_csv(fname, sep=";", engine="python", names=columns, parse_dates=["time"])
    return data


def _get_mode_changes(markers):
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

    ax.plot(acceleration["time"], acceleration["x"], linewidth=1, color="black", alpha=1)
    ax.plot(acceleration["time"], acceleration["y"], linewidth=1, color="#FFCCCE", alpha=1)
    ax.plot(acceleration["time"], acceleration["z"], linewidth=1, color="#86abf9", alpha=1)

    resample_interval = "5s"
    acceleration = acceleration.copy()
    acceleration.index = acceleration["time"]
    acc_median = acceleration.resample(resample_interval).median()

    ax.plot(acc_median["x"], linewidth=2, color="black", label="x - median: "+resample_interval)
    ax.plot(acc_median["y"], linewidth=2, color="red", label="y - median: "+resample_interval)
    ax.plot(acc_median["z"], linewidth=2, color="blue", label="z - median: "+resample_interval)
    ax.legend()

    # acc_min = acceleration.resample("1s").min()
    # acc_max = acceleration.resample("1s").max()
    #
    # ax.plot(acc_max["x"], linewidth=0.5, color="black", alpha=0.9)
    # ax.plot(acc_max["y"], linewidth=0.5, color="red", alpha=0.9)
    # ax.plot(acc_max["z"], linewidth=0.5, color="blue", alpha=0.9)
    #
    # ax.plot(acc_min["x"], linewidth=0.5, color="black", alpha=0.9)
    # ax.plot(acc_min["y"], linewidth=0.5, color="red", alpha=0.9)
    # ax.plot(acc_min["z"], linewidth=0.5, color="blue", alpha=0.9)

    major_ticks = mdates.MinuteLocator(byminute=range(0, 60, 5))
    major_formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(major_ticks)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.set_ylabel(u"Acceleration \n [m/s²]")
    ax.set_title("Acceleration data \n" + title)
    ax.set_ylim(-40, 40)
    if mode_changes is not None:
        for i, mode_change in mode_changes.iterrows():
            ax.axvline(x=mode_change["time"], ymin=-100, ymax=100, color="k", linestyle="--")
            ax.text(s=" "+mode_change["mode"], x=mode_change["time"], y=max_y*0.8, rotation=90,
                    ha='right',
                    bbox=dict(boxstyle='square, pad=2', fc='none', ec='none'))

    return ax


def plot_position(positions, fig=None, subplot_nrRows=1, subplot_nrCols=1, subplot_index=1):

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot_nrRows, subplot_nrCols, subplot_index, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(positions["longitude"], positions["latitude"], color="red", s=5)
    ax.scatter(positions["longitude"][:1], positions["latitude"][:1], color="black", s=100, label="start", marker="x")
    ax.scatter(positions["longitude"][-1:], positions["latitude"][-1:], color="red", s=50, label="end")

    ax.add_image(map_img.Stamen("terrain"),     # Service
                 15,                            # Zoomlevel
                 interpolation='spline36',      # Interpolation (more accurate)
                 )
    ax.legend()
    plt.show()
    return ax


# def plot_acceleration_and_gps(acceleration, positions, mode_changes=None, title=""):
#
#     fig = plt.figure(figsize=(10, 5))
#     ax = plot_position(positions, fig, subplot_nrRows=1, subplot_nrCols=2)
#     ax2 = fig.add_subplot(1, 2, 2)
#     plot_acceleration(acceleration, ax, mode_changes=mode_changes, title=title)
#

#%%


if __name__ == '__main__':
    main()
