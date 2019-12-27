"""
"""

# IMPORTS
import numpy as np
import matplotlib.dates as md
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os

# ===============
# RUN LOCAL
# ===============
# to run colab localy use the menu at the top right and click on "onnect to local runtime"
# if there is a error follow the steps in Troubleshooting (i.e. run console commands in anaconda prompt)

import pandas as pd
import os

# # Johannes Path:
# PATH = r"/home/johannes/Documents/Uni/Data_Mining/Mobility_Task"

# Lukas Path:
PATH = r"C:\Users\feierll\PycharmProjects\UNI\DataMining\Group_Assignments\Mobility Task\white_list"

# Test only
# ----------------------------------
filename = r"10_160_2019-11-14T125859.523/acceleration.csv"
filepath = os.path.join(PATH, filename)
df = pd.read_csv(filepath)
print("test import sucessful")
del filename, filepath, df


# ----------------------------------


# ===============
# READ FILES
# ============

def get_acceleration_data(TRIP):
    """

    :param TRIP:
    :return:
    """
    # IMPORT DATA
    acc_data = pd.read_csv(TRIP + '/acceleration.csv', parse_dates=["time"], index_col="time")

    # Interpolation 100Hz
    # --------------------
    FREQUENCY = "0.01S"
    RESAMPLE =  "0.001S"  # should be higher in precision than the FREQUENCY

    # resamples in higher frequency creating nan, which get interpolated and the resampled with wanted frequency.
    test = acc_data.resample(RESAMPLE).interpolate(freq=FREQUENCY).resample(FREQUENCY).nearest()
    print(test)

    acc_data = test
    # Piecewise Aggregate Approximation
    # --------------------
    # Explanation: https://vigne.sh/posts/piecewise-aggregate-approx/
    paa_freq = "0.1S"
    acc_data = acc_data.resample(paa_freq).mean()

    # remove first and last 30 seconds
    # --------------------
    acc_data = acc_data[300:-300]

    # 2-Norm
    # --------------------
    acc_data["norm"] = np.sqrt(np.square(acc_data).sum(axis=1))

    return acc_data


def get_markers_data(TRIP):
    columns = ["time", "key", "value", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
    markers_data = pd.read_csv(TRIP + '/markers.csv', sep=";", names=columns, skiprows=1, parse_dates=["time"])
    return markers_data


# get all trip names
TRIPS = os.listdir(PATH)

# final dataframe
data = pd.DataFrame()

for trip_directory in TRIPS:

    user_id = trip_directory.split("_")[0]
    trip_id = trip_directory.split("_")[1]
    trip_path = os.path.join(PATH, trip_directory)
    print("TRIP: ", trip_directory)

    # import data
    acc_data = get_acceleration_data(trip_path)
    markers_data = get_markers_data(trip_path)
    pos_data = pd.read_csv(trip_path + '/positions.csv', parse_dates=["time"], index_col="time")

    # # Others:
    # linacc_data = pd.read_csv(trip_path+'/linear_acceleration.csv', parse_dates=["time"], index_col="time")
    # mag_data = pd.read_csv(trip_path+'/magnetic.csv', parse_dates=["time"], index_col="time")
    # ori_data = pd.read_csv(trip_path+'/orientation.csv', parse_dates=["time"], index_col="time")

    # CREATE TRIP-DATA
    # ------------------------------------------------------------------------------------------------------------------
    trip_data = pd.DataFrame()

    # Add acceleration data
    trip_data["x"] = acc_data["x"]
    trip_data["y"] = acc_data["y"]
    trip_data["z"] = acc_data["z"]
    trip_data["norm"] = acc_data["norm"]

    # Add pos data (merging on the trip_data index)
    test = pd.merge_asof(trip_data, pos_data, left_index=True, right_index=True, tolerance=pd.Timedelta("0.1S"))

    # add metadata
    trip_data["id"] = trip_id
    trip_data["uid"] = user_id

    # Add labels
    label_name = "label"
    mode_changes = markers_data[markers_data["key"] == 'CGT_MODE_CHANGED'][["time", "value", "C4"]]
    mode_changes["value"] = pd.to_datetime(mode_changes["value"])

    for i, mode_change in mode_changes.iterrows():
        is_after_change = trip_data.index >= mode_change["value"]
        trip_data.loc[is_after_change, label_name] = mode_change["C4"]
    # ------------------------------------------------------------------------------------------------------------------

    data = data.append(trip_data, ignore_index=True)

data.to_csv(PATH + "/../resulting_data.csv")

import seaborn as sns

sns.pairplot(data, hue="label")
