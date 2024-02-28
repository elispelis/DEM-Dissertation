import sys
sys.path.append("..")

from LaceyClass import LaceyMixingAnalyzer
from data_loader_rnn import RNNLoader
import pandas as pd
import numpy as np
import os


def get_velocity_std(b_coords, div_size, delta_t, Ng, sim_path, bins):

    velocity_path = rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}.csv"

    if not os.path.isfile(velocity_path):

        velocity_means = np.zeros((len(b_coords), 3))
        print("Getting velocities...")    

        for timestep in np.arange(rnn.start_t, rnn.end_t+delta_t, delta_t):
            timestep = rnn.find_nearest(timestep, rnn.deck.timestepValues)
            print(timestep)
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

        np.savetxt(velocity_path, velocity_means, delimiter=",")
    else:
        velocity_means = np.genfromtxt(velocity_path, delimiter=",")
        timestep = rnn.find_nearest(rnn.end_t+delta_t, rnn.deck.timestepValues)
        particles = rnn.get_particle_data(timestep)
        binned_particles = lacey.bin_particles(b_coords, div_size, particles)


    velocity_std_deviation_per_bin = []
    count = 0
    print("Getting velocity stdevs...")

    for bin_velocities, avg_velocity in zip(binned_particles, velocity_means):
        count += 1
        print(f"{count/velocity_means.shape[0]*100:.2f}%")
        if len(bin_velocities) < Ng:
            velocity_std_deviation = [0,0,0]
        else:
            # stdev calculation
            deviations = np.array(bin_velocities[:, 3:6]) - avg_velocity
            squared_deviations = deviations**2
            mean_squared_deviation = np.mean(squared_deviations, axis=0)

            velocity_std_deviation = np.sqrt(mean_squared_deviation)
            
        velocity_std_deviation_per_bin.append(velocity_std_deviation)

    return np.array(velocity_std_deviation_per_bin)

sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
sim_name = sim_names[-1]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
lacey_settings = f"{sim_path[:-4]}_data\Export_Data\Lacey_settings.txt"
velocity_means_path = rf"{sim_path[:-4]}_data\Export_Data\10_5_10.csv"

with open(lacey_settings, 'r') as file:
    preferences = file.readlines()
    minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
    maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
    bins = np.array([int(i) for i in str(preferences[5]).split(',')])
    Ng = float(preferences[7])
    plot = str(preferences[9])
    file.close()
    settings = True

lacey = LaceyMixingAnalyzer(minCoords, maxCoords, bins)
print("Deck loading...")
rnn = RNNLoader(3,4,sim_path)
print("Done")

b_coords, div_size = lacey.grid()
delta_t = 0.05

velocity_std_per_bin = get_velocity_std(b_coords, div_size, delta_t, Ng, sim_path, bins)

np.savetxt(rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}_{int(Ng)}.csv", velocity_std_per_bin, delimiter=",")



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




