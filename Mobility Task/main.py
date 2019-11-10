"""
"""


import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mplleaflet


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


def plot_acceleration(acceleration, mode_changes=None, title=""):

    fig, axes = plt.subplots(3, sharex=True, sharey=True)

    # PLOT all Data
    axes[0].plot(acceleration["time"], acceleration["x"], linewidth=1, color="gray", alpha=1)
    axes[1].plot(acceleration["time"], acceleration["y"], linewidth=1, color="#FFCCCE", alpha=1)
    axes[2].plot(acceleration["time"], acceleration["z"], linewidth=1, color="#86abf9", alpha=1)

    # PLOT median
    resample_interval = "5s"
    acceleration = acceleration.copy()
    acceleration.index = acceleration["time"]
    acc_median = acceleration.resample(resample_interval).median()

    for ax in axes:
        ax.plot(acc_median["x"], linewidth=2, color="black", label="x - median: "+resample_interval)
        ax.plot(acc_median["y"], linewidth=2, color="red", label="y - median: "+resample_interval)
        ax.plot(acc_median["z"], linewidth=2, color="blue", label="z - median: "+resample_interval)
        ax.legend(loc="upper right")

        # FORMATTING
        major_ticks = mdates.MinuteLocator(byminute=range(0, 60, 5))
        major_formatter = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_locator(major_ticks)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.set_ylabel(u"Acceleration \n [m/s²]")
        ax.set_ylim(-40, 40)

    fig.suptitle("Acceleration data \n" + title)

    # PLOT MODE CHANGeS
    for ax in axes:
        max_y = max(acceleration["x"].max(), acceleration["y"].max(), acceleration["z"].max())
        if mode_changes is not None:
            for i, mode_change in mode_changes.iterrows():
                ax.axvline(x=mode_change["time"], ymin=-100, ymax=100, color="k", linestyle="--")
                ax.text(s=" "+mode_change["mode"], x=mode_change["time"], y=max_y*0.7, rotation=90,
                        ha='right',
                        bbox=dict(boxstyle='square, pad=2', fc='none', ec='none'))

    return ax


def plot_position(positions):

    fig, ax = plt.subplots()
    ax.scatter(positions["longitude"], positions["latitude"], color="red", s=5)
    ax.scatter(positions["longitude"][:1], positions["latitude"][:1], color="red", s=100, label="start", marker="X")
    ax.scatter(positions["longitude"][-1:], positions["latitude"][-1:], color="red", s=50, label="end")

    mplleaflet.show()
    return ax

#%%


if __name__ == '__main__':
    main()
