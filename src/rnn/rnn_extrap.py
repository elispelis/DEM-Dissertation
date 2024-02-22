import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from edempy import Deck
from extrapolation import extrapolation
# from LaceyClass import LaceyMixingAnalyzer
# from data_loader_rnn import RNNLoader
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation

sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
sim_name = sim_names[-1]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
id_dict_path = rf"V:\GrNN_EDEM-Sims\{sim_name}_data\Export_Data"
model_paths = ["../../model/model_sl10_tr144.h5", "../../model/model_sl50_tr80.h5", "../../model/model_sl15_tr36.h5", "../../model/model_sl15_tr36_adj.h5", "../../model/model_sl25_tr90_adj.h5", "../../model/model_sl25_tr180_adj.h5", "../../model/model_sl15_tr36_adj_big.h5"]
data_paths = ["../../model/3_4_0.05s.csv", "../../model/3_4_0.01s.csv", "../../model/4_6_0.05s.csv", "../../model/4_6_0.05s_adj.csv", "../../model/3_4_0.01s.csv", "../../model/3_7_0.02s_adj.csv", "../../model/400k_3_5_0.05s_adj.csv"]
case = -1
data_path = data_paths[case]
model_path = model_paths[case]

# # Initialise variables and call class
# simulation = os.path.abspath(os.path.join("..", "..", '..', 'data', "rot_drum", "JKR_periodic_clean", "Rot_drum.dem"))
# sim_path = os.path.dirname(simulation)

#simulation parameters
# start_t = 1
# end_t = 4
# domain_x = (-0.06, 0.06)
# domain_y = (-0.015, 0.015)
# domain_z = (-0.06, 0.06)


# with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../Lacey_settings.txt")), 'r') as file:
#     preferences = file.readlines()
#     minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
#     maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
#     bins = np.array([int(i) for i in str(preferences[5]).split(',')])
#     cut_off = float(preferences[7])
#     plot = str(preferences[9])
#     file.close()
#     settings = True

# print("Loading deck...")
# extrap = extrapolation(start_t, end_t, sim_path, domain_x, domain_y, domain_z, 1, "y")
# print("done")
# lacey = LaceyMixingAnalyzer(minCoords, maxCoords, 5, sim_path)
# rnn = RNNLoader(start_t, end_t, sim_path)

def plot_particles(particle_coords, id_dict, plot, time=None):
    if len(particle_coords[0,:])<6:
        id_color = np.array([id_dict.get(id,0) for id in particle_coords[:,-1]])
        particle_coords = np.column_stack((particle_coords, id_color))
        particle_coords = particle_coords[particle_coords[:,1].argsort()]

    if plot == True:
        fig, ax = plt.subplots(figsize=(8,8))

        ax.add_patch(plt.Circle((0,0), 0.07, color="lightblue"))
        ax.set_ylim(-0.08, 0.03)
        ax.set_xlim(-0.08, 0.08)

        r = 0.0005
        # radius in display coordinates:
        r_ = ax.transData.transform([r,0])[0] - ax.transData.transform([0,0])[0]
        # marker size as the area of a circle
        particle_plt_size = np.pi * r_**2

        ax.scatter(particle_coords[:,0], particle_coords[:,2], s=particle_plt_size, linewidth=0.25, c=particle_coords[:,-1], cmap="coolwarm")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{time:.2f}s")
        
        plt.show()

    return particle_coords

def import_dict(dict_path, dict_name):
    for file in os.listdir(dict_path):
        if file.startswith(dict_name):
            pos_dict_name = file
            with open(os.path.join(dict_path, pos_dict_name), "rb") as file:
                pos_dict = pickle.load(file)
    return pos_dict


def fix_particle_coords(local_mean, drum_r, drum_w):
    
    distances = np.sqrt((local_mean[:, 0])**2+(local_mean[:, 2])**2)
    violating_particles = np.where(distances>drum_r)

    profile_fixed = len(violating_particles)

    for particle_idx in violating_particles:
        # Normalize the position vector
        norm_factor = drum_r / distances[particle_idx]
        local_mean[particle_idx, 0] *= norm_factor
        local_mean[particle_idx, 2] *= norm_factor
    
    violating_particles = np.where(abs(local_mean[:,1])>drum_w)

    side_fixed = len(violating_particles)

    for particle_idx_w in violating_particles:
            sign_w = np.sign(local_mean[particle_idx_w, 1])
            local_mean[particle_idx_w, 1] = sign_w * drum_w
        
    print(f"Fixed {profile_fixed+side_fixed} particle. (Profile:{profile_fixed}, Side: {side_fixed}" )

    return local_mean


# def bin_particles(particles, bounding_box, num_bins_per_dimension):
#     Extract particle coordinates
#     particle_coordinates = particles[:, :3]

#     Create 3D grid representing the bounding box
#     bins = [np.linspace(bounding_box[i][0], bounding_box[i][1], num_bins_per_dimension[i] + 1) for i in range(3)]

#     Digitize particle coordinates to assign them to bins
#     bin_indices = [np.digitize(particle_coordinates[:, i], bins[i]) - 1 for i in range(3)]

#     Combine bin indices into a single index for each particle
#     particle_bin_indices = bin_indices[0] + num_bins_per_dimension[0] * (bin_indices[1] + num_bins_per_dimension[1] * bin_indices[2])

#     Create a 3D array to store particles in each bin
#     max_bin_index = np.max(particle_bin_indices)
#     particle_bins = np.empty((max_bin_index + 1,), dtype=object)

#     Initialize each element as an empty list
#     for i in range(len(particle_bins)):
#         particle_bins[i] = []

#     Populate the array with particles
#     for i, bin_index in enumerate(particle_bin_indices):
#         particle_bins[bin_index].append(particles[i])

#     return particle_bins


# bounding_box = [(-0.07, 0.07), (-0.025, 0.025), (-0.07, 0.07)]
# num_bins_per_dimension = [5, 5, 5]

# particles_in_bins = bin_particles(rnn.get_particle_data(3), bounding_box, num_bins_per_dimension)

# b_coords, div_size = lacey.grid()
# particles_in_bins = lacey.bin_particles(b_coords, div_size, rnn.get_particle_data(start_t)[:, :3])

id_dict = import_dict(id_dict_path, "id_dict")

model = tf.keras.models.load_model(model_path)
df = pd.read_csv(data_path, index_col=0)


num_features = 3
num_timesteps = df.shape[1] // num_features
num_particles = df.shape[0]
seq_length = 15

last_seq = df.iloc[:, (num_timesteps-seq_length)*num_features:]

last_seq = last_seq.values.reshape(-1, seq_length, num_features)

particle_loc_fix = False

# last_seq = df.iloc[:, (num_timesteps-seq_length)*num_features:]
# index_values = last_seq.index.values.reshape(-1, 1, 1)
# index_values_for_last_elem = np.tile(index_values, (1, seq_length, 1))
# last_seq = last_seq.values.reshape(-1, seq_length, num_features)
# last_seq = np.concatenate([last_seq, index_values_for_last_elem], axis=2)

start_t = 4.9
t_rnn = 0.05
extrap_time = t_rnn*200
drum_r = 0.07
drum_w = 0.025


#distance = (last_seq[:, -1, :][:, 0])**2+(last_seq[:, -1, :][:, 2])**2

for i in range(int(extrap_time/t_rnn)):
    pred_timestep = model.predict(last_seq)

    if particle_loc_fix == True:
        pred_timestep = fix_particle_coords(pred_timestep, drum_r, drum_w)
    
    last_seq = last_seq[:, 1:, :]
    last_seq = np.concatenate((last_seq, pred_timestep[:, np.newaxis, :]), axis=1)

    if i % 5 == 0 and i != 0:
        id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
        pred_timestep = np.hstack((pred_timestep, id_column))
        time_i = start_t+(i+1)*t_rnn
        print(f"{time_i}s")
        plot_particles(pred_timestep, id_dict, True, time_i)