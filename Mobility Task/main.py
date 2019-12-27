"""
"""


import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mplleaflet


def main():

    # PARAMETER
    # --------------------------------------------
    data_directory = r"./white_list"  # local path
    subdir_list = os.listdir(data_directory)
    # --------------------------------------------

    total_data = pd.DataFrame()
    label_name = "label"

    for subdir in subdir_list:
        subdir = subdir_list[3]
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

        break

        # # ==================
        # # MAIN
        # # ==================
        #
        # # DATAPOINTS
        # # --------------------------
        # data = pd.DataFrame()
        # acceleration = acceleration[["x", "y", "z"]]
        # # data[["std_x", "std_y", "std_z"]] = acceleration.resample("10s").std()      # Standard derivative
        # data["std_sum"] = acceleration.resample("10s").std().sum(axis=1)      # Standard derivative
        #
        # # data[["max_x", "max_y", "max_z"]] = acceleration.resample("10s").max()      # Maximum
        # # data[["min_x", "min_y", "min_z"]] = acceleration.resample("10s").min()      # Minimum
        # # data[["abs_x", "abs_y", "abs_z"]] = acceleration.abs().resample("10s").max()     # Absolute Max/Min
        # data["absmax_sum"] = acceleration.abs().resample("10s").max().sum(axis=1)     # Absolute Max/Min
        # # data[["abs_max_x", "abs_max_y", "abs_max_z"]] = acceleration.abs().resample("10s").max()     # Absolute Max/Min
        #
        # # data[["sum_x", "sum_y", "sum_z"]] = acceleration.resample("10s").sum()           # sum
        # # data["sum_sum"] = acceleration.abs().resample("10s").sum().sum(axis=1)  # sum
        #
        # # data[["median_x", "median_y", "median_z"]] = acceleration.resample("10s").median()  # median
        # # data["median_sum"] = acceleration.abs().resample("10s").median().sum(axis=1)  # median
        #
        #
        # data[label_name] = "NONE"
        # for mode_change in mode_changes.iterrows():
        #     print(mode_change)
        #     # TODO: Check if the right mode_change tst was used
        #     data.loc[data.index >= mode_change[1]["datetime"], label_name] = mode_change[1]["mode"]
        #
        # # Append to other data
        # total_data = total_data.append(data)

    # import seaborn as sns
    # sns.pairplot(total_data, hue=label_name)

    return True


#%%

def _read_position_csv(dir):
    fname = os.path.join(dir, "positions.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    data.index = data.time
    return data


def _read_acceleration_csv(dir):
    fname = os.path.join(dir, "acceleration.csv")
    data = pd.read_csv(fname, parse_dates=["time"])
    data.index = data.time
    return data


def _read_markers_csv(dir):
    fname = os.path.join(dir, "markers.csv")
    columns = ["time", "key", "value", "value1", "value2", "value3", "value4", "value5", "value6", "value7", "value8"]
    data = pd.read_csv(fname, sep=";", engine="python", names=columns, parse_dates=["time"])
    data.index = data.time
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
    fig.suptitle("Acceleration data \n" + title)

    # PLOT  Acceleration data
    axes[0].plot(acceleration["time"], acceleration["x"], linewidth=1, color="gray")
    axes[1].plot(acceleration["time"], acceleration["y"], linewidth=1, color="#FFCCCE")
    axes[2].plot(acceleration["time"], acceleration["z"], linewidth=1, color="#86abf9")

    # FORMATTING
    for ax in axes:
        ax.axhline(color="k", linestyle="--", alpha=0.2)             # Lines/Grid for easier comparison
        ax.axhline(y=20, color="k", linestyle=":", alpha=0.2)
        ax.axhline(y=-20, color="k", linestyle=":", alpha=0.2)
        major_ticks = mdates.MinuteLocator(byminute=range(0, 60, 5))    # Set Locator to 5Minute steps
        major_formatter = mdates.DateFormatter('%H:%M')                 # Format what to display
        ax.xaxis.set_major_locator(major_ticks)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.set_ylabel(u"Acceleration \n [m/s²]")
        ax.set_ylim(-40, 40)

    # PLOT Medians
    resample_interval = "5s"
    acceleration.index = acceleration["time"]
    acc_median = acceleration.resample(resample_interval).median()
    axes[0].plot(acc_median["x"], linewidth=2, color="black", label="x - median:" + resample_interval)  # Medians
    axes[1].plot(acc_median["y"], linewidth=2, color="red", label="y - median:" + resample_interval)
    axes[2].plot(acc_median["z"], linewidth=2, color="blue", label="z - median:" + resample_interval)
    fig.legend()

    # PLOT MODE CHANGES
    for ax in axes:
        if mode_changes is not None:
            for i, mode_change in mode_changes.iterrows():
                ax.axvline(x=mode_change["time"], color="k", linestyle="--")        # Draw Line
                ax.text(s=mode_change["mode"], x=mode_change["time"],               # Draw Annotation
                        y=20, rotation=90, ha='right',
                        )

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
