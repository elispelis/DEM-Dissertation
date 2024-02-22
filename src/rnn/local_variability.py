from edempy import Deck
import sys
sys.path.append("..")

from LaceyClass import LaceyMixingAnalyzer
from data_loader_rnn import RNNLoader
import pandas as pd
import numpy as np
import os

sim_names = ["Rot_drum_mono.dem", "Rot_drum_binary_mixed.dem"]
sim_name = sim_names[0]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}"
lacey_settings = f"{sim_path[:-4]}_data\Export_Data\Lacey_settings.txt"

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
