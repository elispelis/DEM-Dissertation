#import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from edempy import Deck
from extrapolation import extrapolation
from LaceyClass import LaceyMixingAnalyzer
from data_loader_rnn import RNNLoader
import os

sim_names = ["Rot_drum_mono.dem", "Rot_drum_binary_mixed.dem"]
sim_name = sim_names[0]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}"
model_path = "../model/model_sl10_tr144.h5"
data_path = "../model/3_4_0.05s.csv"

# Initialise variables and call class
simulation = os.path.abspath(os.path.join("..", "..", '..', 'data', "rot_drum", "JKR_periodic_clean", "Rot_drum.dem"))
sim_path = os.path.dirname(simulation)

#simulation parameters
start_t = 1
end_t = 4
domain_x = (-0.06, 0.06)       
domain_y = (-0.015, 0.015)
domain_z = (-0.06, 0.06)
num_bins = 1
direction = "y"

with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../Lacey_settings.txt")), 'r') as file:
    preferences = file.readlines()
    minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
    maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
    bins = np.array([int(i) for i in str(preferences[5]).split(',')])
    cut_off = float(preferences[7])
    plot = str(preferences[9])
    file.close()
    settings = True


extrap = extrapolation(start_t, end_t, simulation, domain_x, domain_y, domain_z, num_bins, direction)
lacey = LaceyMixingAnalyzer(minCoords, maxCoords, 5, simulation)
rnn = RNNLoader(start_t, end_t, simulation)

b_coords, div_size = lacey.grid()
particles_in_bins = lacey.bin_particles(b_coords, div_size, rnn.get_particle_data(start_t)[:, :3])




# model = tf.keras.models.load_model(model_path)
# df = pd.read_csv(data_path, index_col=0)


# num_features = 3
# num_timesteps = df.shape[1] // num_features
# num_particles = df.shape[0]
# seq_length = 10

# last_seq = df.iloc[:, (num_timesteps-seq_length)*num_features:]

# last_seq = last_seq.values.reshape(-1, seq_length, 3)

# t_rnn = 0.05
# extrap_time = t_rnn


# for i in range(int(extrap_time/t_rnn)):
#     pred_timestep = model.predict(last_seq)
#     last_seq = last_seq[:, 1:, :]
#     last_seq = np.concatenate((last_seq, pred_timestep[:, np.newaxis, :]), axis=1)