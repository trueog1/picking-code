import numpy as np
import rosbag
from geometry_msgs.msg import Wrench
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import csv 
from Ag_functions import bag_to_csv, butter_lowpass_filter, total_time, match_times, flipping_data, bag_pressure, pressure_time, elapsed_time, filter_force_m
from UR5e import UR5e_ros1
import scipy as scipy
from kneed import KneeLocator

number = 3
attempt = 1
actual = f'Success'
name = f'2023111_realapple{number}_mode_dual_attempt_{attempt}_orientation_0_yaw_0' #2023111_realapple20_mode_dual_attempt_2_orientation_0_yaw_0

file = bag_to_csv(name)

data = np.loadtxt('./' + file + '.csv', dtype = "float", delimiter = ',')

#current guesstimation for x values - sample number
sec = data[:,-1]
nsec = data[:,-2]
force =  data[:, :-2]

time = sec.tolist()
ntime = nsec.tolist()

#normalize data
f_arr = np.linalg.norm(force, axis=1)

tot_time = total_time(time, ntime)
etime_force = elapsed_time(tot_time, tot_time)

#displacement time baby
path = UR5e_ros1(0,name)    
total_disp = path.z
joint_times_sec = path.times_seconds
joint_times_nsec = path.times_nseconds

tot_time_j = total_time(joint_times_sec, joint_times_nsec)
etime_joint = elapsed_time(tot_time_j, tot_time_j)

pressure_array = bag_pressure(file)   
p_arr = np.linalg.norm(pressure_array, axis=1)

etimes_pressure = pressure_time(file)

f_arr_col = f_arr[..., None]
total_disp_col = np.array(total_disp)[..., None]
p_arr_col = p_arr[..., None]

final_force, delta_x, general_time = match_times(etime_force, etime_joint,f_arr_col, total_disp_col)

final_pressure, p_dis, other_time = match_times(etimes_pressure, etime_joint,p_arr_col, total_disp_col)

fp = final_pressure.tolist()

filtered = filter_force_m(final_force,21)

#Central Difference
central_diff = []
h = 50         # 2 * delta-t value
for i in range(h, (len(filtered))):
    diff = (filtered[i] - filtered[i-h])/ (h)
    central_diff.append(diff)
    i += 1

# Filter requirements.
fs = 500.0       # sample rate, Hz
cutoff = 50      # desired cutoff frequency of the filter, Hz 
order = 2       # sin wave can be approx represented as quadratic

low_cdiff = butter_lowpass_filter(central_diff, cutoff, fs, order)

low_delta_x = scipy.signal.savgol_filter(delta_x[:,0], 600, 1)

#selecteding the correct data to use
if number != 25:
    kn = KneeLocator(general_time, low_delta_x, curve='concave', direction='decreasing') 
    kn1 = KneeLocator(general_time, low_delta_x, curve='convex', direction='decreasing') 
    #idx1 = kn.minima_indices[0]
    idx2 = kn1.minima_indices[0]
    idx = kn1.minima_indices[-2]
    peaks, peak_index = scipy.signal.find_peaks(low_delta_x, height = 0.03, plateau_size = (None, None))
    peak = int(peak_index['right_edges'][-1])
    peak1 = int(peak_index['left_edges'][0])

    #pressure turn
    pturn1 = np.where(final_pressure == np.min(final_pressure))[0]           

    #turn = [i for i, e in enumerate(low_delta_x) if e < (low_delta_x[peak]-0.16) and i > peak] 
    idx2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0][0]
    turn = np.where(low_delta_x == np.min(low_delta_x[idx2:-1]))[0] #this is not picking where the turn around happens exactly
    turn2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]

    cropped_x = delta_x[idx2 + 500:turn[0]]
    new_x_part = flipping_data(cropped_x)

else:
    idx2 = np.where(low_delta_x == np.min(low_delta_x))[0][0]
    turn = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]
    new_x_part = delta_x[idx2 + 500:turn[0]]