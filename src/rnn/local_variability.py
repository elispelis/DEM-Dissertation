import os
import sys

import numpy as np

sys.path.append("..")
from gridbin import GridBin
from data_loader_rnn import RNNLoader


def get_velocity_std(delta_t, drum_radius, sim_path, bins, minCoords, maxCoords):
    velocity_path = rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}.npy"
    sr_grid_bin = GridBin(minCoords, maxCoords, *bins)

    if not os.path.isfile(velocity_path):

        velocity_means = np.zeros((sr_grid_bin.numBins, 3))
        print("Getting velocities...")

        timesteps = np.arange(rnn.start_t, rnn.end_t + delta_t, delta_t)

        for t in timesteps:
            timestep = rnn.find_nearest(t, rnn.deck.timestepValues)
            print(timestep)
            particles = rnn.get_particle_data(timestep)

            mean_velocities, mean_fluctuating_velocities, binned_indxs = sr_grid_bin.calculate_velocity_stdev(
                particles.position, particles.velocity, drum_radius)
            velocity_means += mean_velocities

        velocity_means /= timesteps.size

        with open(velocity_path, 'wb') as f:
            np.save(f, velocity_means)

    else:
        timestep = rnn.find_nearest(rnn.end_t + delta_t, rnn.deck.timestepValues)
        particles = rnn.get_particle_data(timestep)
        mean_velocities, mean_fluctuating_velocities, binned_indxs = sr_grid_bin.calculate_velocity_stdev(
            particles.position, particles.velocity, drum_radius)

    return mean_fluctuating_velocities


sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
sim_name = sim_names[-1]
sim_path = rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
lacey_settings = f"{sim_path[:-4]}_data\Export_Data\Lacey_settings_SR.txt"
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

print("Deck loading...")
rnn = RNNLoader(3, 4, sim_path)
print("Done")

delta_t = 0.05

drum_radius = 0.07
velocity_std_per_bin = get_velocity_std(delta_t, drum_radius, sim_path, bins, minCoords, maxCoords)

with open(rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}_{int(Ng)}.npy", 'wb') as f:
    np.save(f, velocity_std_per_bin)
