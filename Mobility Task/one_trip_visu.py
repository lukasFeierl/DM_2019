import numpy as np
import matplotlib.dates as md
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os


# to run colab localy use the menu at the top right and click on "onnect to local runtime"
# if there is a error follow the steps in Troubleshooting (i.e. run console commands in anaconda prompt)

import pandas as pd
import os

# Johannes Path:
PATH = r"/home/johannes/Documents/Uni/Data_Mining/Mobility_Task"

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



# Select Data
# --------------
def get_trip_paths(user_id=None, trip_id=None, stud2="/cgt-stud_2/", stud1="/cgt-stud_1/"):
    """ returns the paths of all trips with the user_id and trip id inserted"""
    # Get all dirs in the directories
    TRIPS_1 = os.listdir(PATH + stud1)
    TRIPS_2 = os.listdir(PATH + stud2)

    # Filter for User ID
    if user_id is not None:
        TRIPS_1 = [x for x in TRIPS_1 if x.split("_")[0] == str(user_id)]
        TRIPS_2 = [x for x in TRIPS_2 if x.split("_")[0] == str(user_id)]

    # Filter for trip id
    if trip_id is not None:
        TRIPS_1 = [x for x in TRIPS_1 if x.split("_")[1] == str(trip_id)]
        TRIPS_2 = [x for x in TRIPS_2 if x.split("_")[1] == str(trip_id)]

    # Add PATH and directories to path
    TRIPS_1 = [PATH + stud1 + x for x in TRIPS_1]
    TRIPS_2 = [PATH + stud2 + x for x in TRIPS_2]

    # Combine Data
    TRIPS = TRIPS_1 + TRIPS_2
    if TRIPS == []:
        raise ValueError("No trip found with userid: "+str(user_id) +" and trip id: "+str(trip_id))
    return TRIPS

# ALTERNATIVE I
# --------------
user_id = 26
trip_id = None
TRIPS = get_trip_paths(user_id=user_id, trip_id=trip_id)
TRIP = TRIPS[0]


# ALTERNATIVE II
# --------------
stud = "/cgt-stud_2/"
TRIPS = ['26_127_2019-11-18T080954.995',
       '26_128_2019-11-19T084727.905',
       '26_129_2019-11-18T122510.838',
       '26_131_2019-11-19T135849.719',
       '26_165_2019-12-05T071013.538']
TRIP = [PATH + stud + x for x in TRIPS]
TRIP = TRIPS[1]


# Read data
# ----------
acc_data = pd.read_csv(TRIP + '/acceleration.csv')
act_data = pd.read_csv(TRIP + '/activity_records.csv')
# gta_data = pd.read_csv(TRIP+'/gt_annotations_web.csv')
linacc_data = pd.read_csv(TRIP + '/linear_acceleration.csv')
mag_data = pd.read_csv(TRIP + '/magnetic.csv')
mark_data = pd.read_csv(TRIP + '/markers.csv', sep=';', names=["C%s" % i for i in range(1, 11)], skiprows=1, parse_dates=["C1"])
ori_data = pd.read_csv(TRIP + '/orientation.csv')
pos_data = pd.read_csv(TRIP + '/positions.csv')


# Modify Marker
markers = mark_data['C4'][mark_data['C2']=='CGT_MODE_CHANGED'].values
markers_times = mark_data['C1'][mark_data['C2']=='CGT_MODE_CHANGED'].values
start = mark_data["C1"][mark_data['C3']=='START'].values[0]
stop = mark_data["C1"][mark_data['C3']=='STOP'].values[0]

print("total trip time:", round((stop-start)/60, 2), "minutes")