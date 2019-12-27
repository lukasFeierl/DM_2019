

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
PATH = r"C:\Users\feierll\PycharmProjects\UNI\DataMining\Group_Assignments\Mobility Task"


# Test only
# ----------------------------------
filename = r"cgt-stud_1/2_1_2019-10-18T094134.467/acceleration.csv"
filepath = os.path.join(PATH, filename)
df = pd.read_csv(filepath)
print("test import sucessful")
del filename, filepath, df
# ----------------------------------



# ===============
# IMPORT DATA
# ===============
# TODO: Convention unklar Leerzeichen vor oder nach "=" ?


# Alternative I
# --------------
# Directory "white_list"
whitelist_dir = "/white_list/"
TRIPS = os.listdir(PATH + whitelist_dir)
TRIPS = [PATH + whitelist_dir + TRIP for TRIP in TRIPS]


# # Alternative II
# # --------------
# stud1 = "/cgt-stud_1/"
# stud2 = "/cgt-stud_2/"
#
# # Get all Trips in the list
# TRIPS_1 = os.listdir(PATH + stud1)
# TRIPS_1 = [PATH + stud1 + TRIP for TRIP in TRIPS_1]
#
# TRIPS_2 = os.listdir(PATH + stud2)
# TRIPS_2 = [PATH + stud2 + TRIP for TRIP in TRIPS_2]
# TRIPS = TRIPS_1 + TRIPS_2


for TRIP in TRIPS:

    # Read data
    acc_data = pd.read_csv(TRIP+'/acceleration.csv', parse_dates=["time"])
    act_data = pd.read_csv(TRIP+'/activity_records.csv')
    #gta_data = pd.read_csv(TRIP+'/gt_annotations_web.csv')
    linacc_data = pd.read_csv(TRIP+'/linear_acceleration.csv')
    mag_data = pd.read_csv(TRIP+'/magnetic.csv')
    mark_data = pd.read_csv(TRIP+'/markers.csv', sep=';', names=["C%s" % i for i in range(1, 11)])
    ori_data = pd.read_csv(TRIP+'/orientation.csv')
    pos_data = pd.read_csv(TRIP+'/positions.csv')

# # CREATE TRIP-DATA
    # # ------------------------------------------------------------------------------------------------------------------
    # TRIP_SEGMENTS = "10S"
    #
    # new_data = pd.DataFrame()
    #
    # def is_equal(array_like):
    #     if (array_like[0] != array_like[1:]).any():
    #         print("found a NAN")
    #         return np.NaN
    #     return array_like[0]
    #
    # new_data["label"] = acc_data["label"].resample(TRIP_SEGMENTS).apply(is_equal)
    # new_data["median"] = acc_data["norm"].resample(TRIP_SEGMENTS).median()
    # new_data["maximum"] = acc_data["norm"].resample(TRIP_SEGMENTS).max()
    # new_data["std"] = acc_data["norm"].resample(TRIP_SEGMENTS).std()
    #
    # # new_data["id"] = trip_id
    #
    # new_data = new_data.dropna()
    # # ------------------------------------------------------------------------------------------------------------------
