from edempy import Deck
import sys
sys.path.append("..")

from LaceyClass import LaceyMixingAnalyzer
from data_loader_rnn import RNNLoader
import pandas as pd
import numpy as np
import os


def get_velocity_std(b_coords, div_size, delta_t):

    velocity_means = np.zeros((len(b_coords), 3))    

    for timestep in np.arange(rnn.start_t, rnn.end_t+delta_t, delta_t):
        timestep = rnn.find_nearest(timestep, rnn.deck.timestepValues)
        particles = rnn.get_particle_data(timestep)
        binned_particles = lacey.bin_particles(b_coords, div_size, particles)

        for i, bin in enumerate(binned_particles):
            cur_bin = np.array(bin)
            if cur_bin.shape[0] != 0:
                mean_x_vel = np.mean(cur_bin[:, 3])
                mean_y_vel = np.mean(cur_bin[:, 4])
                mean_z_vel = np.mean(cur_bin[:, 5])
            else: 
                mean_x_vel = mean_y_vel = mean_z_vel = 0

            velocity_means[i, 0] += mean_x_vel
            velocity_means[i, 1] += mean_y_vel
            velocity_means[i, 2] += mean_z_vel

    velocity_means /= (rnn.end_t - rnn.start_t) / delta_t

    velocity_std_deviation_per_bin = []

    for bin_velocities, avg_velocity in zip(binned_particles, velocity_means):
        if len(bin_velocities) == 0:
            velocity_std_deviation = [0,0,0]
        else:
            # Step 1: Calculate Deviation from Mean
            deviations = np.array(bin_velocities[:, 3:6]) - avg_velocity
            
            # Step 2: Square the Deviations
            squared_deviations = deviations**2
            
            # Step 3: Calculate the Mean of Squared Deviations
            mean_squared_deviation = np.mean(squared_deviations, axis=0)
            
            # Step 4: Calculate the Square Root
            velocity_std_deviation = np.sqrt(mean_squared_deviation)
            
        velocity_std_deviation_per_bin.append(velocity_std_deviation)

    return np.array(velocity_std_deviation_per_bin)

sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
sim_name = sim_names[-1]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
lacey_settings = f"{sim_path}_data\Export_Data\Lacey_settings.txt"
velocity_means_path = rf"{sim_path}_data\Export_Data\10_5_10.csv"

with open(lacey_settings, 'r') as file:
    preferences = file.readlines()
    minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
    maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
    bins = np.array([int(i) for i in str(preferences[5]).split(',')])
    cut_off = float(preferences[7])
    plot = str(preferences[9])
    file.close()
    settings = True

lacey = LaceyMixingAnalyzer(minCoords, maxCoords, bins)
print("Deck loading...")
rnn = RNNLoader(3,4,sim_path)
print("Done")

b_coords, div_size = lacey.grid()
cut_off = 0.0001
delta_t = 0.05

velocity_std_per_bin = get_velocity_std(b_coords, div_size, delta_t)

np.savetxt(rf"{sim_path}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}.csv", velocity_std_per_bin, delimiter=",")



# velocity_means = np.genfromtxt(velocity_means_path, delimiter=",")

# def velocity_std(velocity_means, binned_particles):
#     velocity_std_deviation_per_bin = []

#     for bin_velocities, avg_velocity in zip(binned_particles, velocity_means):
#         if len(bin_velocities) == 0:
#             velocity_std_deviation = [0,0,0]
#         else:
#             # Step 1: Calculate Deviation from Mean
#             deviations = np.array(bin_velocities[:, 3:6]) - avg_velocity
            
#             # Step 2: Square the Deviations
#             squared_deviations = deviations**2
            
#             # Step 3: Calculate the Mean of Squared Deviations
#             mean_squared_deviation = np.mean(squared_deviations, axis=0)
            
#             # Step 4: Calculate the Square Root
#             velocity_std_deviation = np.sqrt(mean_squared_deviation)
            
#         velocity_std_deviation_per_bin.append(velocity_std_deviation)

#     return np.array(velocity_std_deviation_per_bin)




