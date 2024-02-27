import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from edempy import Deck
from extrapolation import extrapolation
from LaceyClass import LaceyMixingAnalyzer
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation

sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
sim_name = sim_names[-1]
sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
id_dict_path = rf"V:\GrNN_EDEM-Sims\{sim_name}_data\Export_Data"
model_paths = ["../../model/model_sl10_tr144.h5", "../../model/model_sl50_tr80.h5", "../../model/model_sl15_tr36.h5", "../../model/model_sl15_tr36_adj.h5", "../../model/model_sl25_tr90_adj.h5", "../../model/model_sl25_tr180_adj.h5", "../../model/model_sl15_tr36_adj_big.h5"]
data_paths = ["../../model/3_4_0.05s.csv", "../../model/3_4_0.01s.csv", "../../model/4_6_0.05s.csv", "../../model/4_6_0.05s_adj.csv", "../../model/3_4_0.01s.csv", "../../model/3_7_0.02s_adj.csv", "../../model/Rot_drum_400k_3_5_0.05s_adj.csv"]
case = -1
data_path = data_paths[case]
model_path = model_paths[case]

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

    violating_particles = np.where(abs(local_mean[:,1])>drum_w)

    side_fixed = len(violating_particles[0])

    for particle_idx_w in violating_particles:
            #print(local_mean[particle_idx_w])
            sign_w = np.sign(local_mean[particle_idx_w, 1])
            local_mean[particle_idx_w, 1] = sign_w * drum_w

    distances = np.sqrt((local_mean[:, 0])**2+(local_mean[:, 2])**2)
    violating_particles = np.where(distances>drum_r)

    profile_fixed =  len(violating_particles[0])

    for particle_idx in violating_particles:
        # Normalize the position vector
        #print(local_mean[particle_idx])
        norm_factor = drum_r / distances[particle_idx]
        local_mean[particle_idx, 0] *= norm_factor
        local_mean[particle_idx, 2] *= norm_factor
    
        
    print(f"Fixed {profile_fixed+side_fixed} particle(s). (Profile: {profile_fixed}, Side: {side_fixed})" )

    return local_mean

def generate_random_velocity(std_deviation):
    x_vel_random = np.random.normal(loc=0, scale=std_deviation[0])
    y_vel_random = np.random.normal(loc=0, scale=std_deviation[1])
    z_vel_random = np.random.normal(loc=0, scale=std_deviation[2])

    return np.array((x_vel_random, y_vel_random, z_vel_random))

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
stochastic_random = True

if stochastic_random == True:    
    np.random.seed(42)
    particle_loc_fix = True

    lacey_settings = f"{sim_path[:-4]}_data\Export_Data\Lacey_settings.txt"

    with open(lacey_settings, 'r') as file:
        preferences = file.readlines()
        minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
        maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
        bins = np.array([int(i) for i in str(preferences[5]).split(',')])
        Ng = int(preferences[7])
        plot = str(preferences[9])
        file.close()
        settings = True

    velocity_std_path = rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}_{Ng}.csv"

    lacey = LaceyMixingAnalyzer(minCoords, maxCoords, bins)

    b_coords, div_size = lacey.grid()
    velocity_stds = np.genfromtxt(velocity_std_path, delimiter=",")

start_t = 4.9
t_rnn = 0.05
extrap_time = t_rnn*200
drum_r = 0.07
drum_w = 0.025


#distance = (last_seq[:, -1, :][:, 0])**2+(last_seq[:, -1, :][:, 2])**2

for i in range(int(extrap_time/t_rnn)):
    pred_timestep = model.predict(last_seq)
    id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
    pred_timestep = np.hstack((pred_timestep, id_column))

    if stochastic_random == True:
        
        #fix particles to grid
        pred_timestep = fix_particle_coords(pred_timestep, drum_r, drum_w)

        #bin particles and apply velocity_std while tracking id
        binned_particles = lacey.bin_particles(b_coords, div_size, pred_timestep)
        #break
        for bin_velocities, velocity_std in zip(binned_particles, velocity_stds):
            if len(bin_velocities) == 0: #maybe velocity_std instead?
                continue
            else:
                for j in range(len(bin_velocities)):
                    bin_velocities[j, :3] += generate_random_velocity(velocity_std)*t_rnn
        
        binned_particles = np.vstack(binned_particles)
        sorted_indices = np.argsort(binned_particles[:, -1])

        # Sort the array using the obtained indices
        pred_timestep = binned_particles[sorted_indices][:,:3]



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