import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append("..")

from LaceyClass import LaceyMixingAnalyzer
import os
import matplotlib.pyplot as plt
import pickle
from time import time
from gridbin import GridBin
from helpers import Lacey, fix_particle_coords, unpack_mixing_results


def plot_particles(particle_coords, id_dict, plot, time, **kwargs):
    plot_path = None


    for key, value in kwargs.items():
        plot_path = value
    
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

        if plot_path:
            fig.savefig(plot_path)
            plt.clf()
            plt.close(fig)


    return particle_coords

def import_dict(dict_path, dict_name):
    for file in os.listdir(dict_path):
        if file.startswith(dict_name):
            pos_dict_name = file
            with open(os.path.join(dict_path, pos_dict_name), "rb") as file:
                pos_dict = pickle.load(file)
    return pos_dict

if __name__ == "__main__":
    sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
    sim_name = sim_names[-1]
    sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
    id_dict_path = rf"V:\GrNN_EDEM-Sims\{sim_name}_data\Export_Data"
    model_paths = ["../../model/model_sl10_tr144.h5", "../../model/model_sl50_tr80.h5", "../../model/model_sl15_tr36.h5", "../../model/model_sl15_tr36_adj.h5", "../../model/model_sl25_tr90_adj.h5", "../../model/model_sl25_tr180_adj.h5", "../../model/model_sl15_tr36_adj_big.h5"]
    data_paths = ["../../model/3_4_0.05s.csv", "../../model/3_4_0.01s.csv", "../../model/4_6_0.05s.csv", "../../model/4_6_0.05s_adj.csv", "../../model/3_4_0.01s.csv", "../../model/3_7_0.02s_adj.csv", "../../model/Rot_drum_400k_3_5_0.05s_adj.csv"]
    case = -1
    data_path = data_paths[case]
    model_path = model_paths[case]

    #load id dictionary, model and starting series
    id_dict = import_dict(id_dict_path, "id_dict")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')

    df = pd.read_csv(data_path, index_col=0)
    num_features = 3
    num_timesteps = df.shape[1] // num_features
    num_particles = df.shape[0]
    seq_length = 15

    last_seq = df.iloc[:, (num_timesteps-seq_length)*num_features:]
    last_seq = last_seq.values.reshape(-1, seq_length, num_features)

    #extrapolation settings
    particle_loc_fix = False
    stochastic_random = True
    track_lacey = True
    save_plots = True
    show_plots = True

    start_t = 4.9
    t_rnn = 0.05
    end_t = 20
    extrap_time = end_t - start_t
    drum_r = 0.07
    drum_w = 0.025

    if stochastic_random == True:    
        np.random.seed(42)
        particle_loc_fix = True

        lacey_settings = f"{sim_path[:-4]}_data\Export_Data\Lacey_settings_SR.txt"

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

        #b_coords, div_size = lacey.grid()
        sr_grid_bin = GridBin(minCoords, maxCoords, *bins)
        velocity_stds = np.genfromtxt(velocity_std_path, delimiter=",")

    if save_plots == True:
        show_plots = False
        plots_path = rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}_{Ng}_plots_new"
        os.makedirs(plots_path, exist_ok=True)

    if track_lacey == True:    

        lacey_settings = f"{sim_path[:-4]}_data\Export_Data\Lacey_settings.txt"

        with open(lacey_settings, 'r') as file:
            preferences = file.readlines()
            minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
            maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
            bins = np.array([int(i) for i in str(preferences[5]).split(',')])
            cut_off = int(preferences[7])
            plot = str(preferences[9])
            file.close()
            settings = True

        lacey = LaceyMixingAnalyzer(minCoords, maxCoords, bins)

        b_coords_lacey, div_size_lacey = lacey.grid()
        extrapolated_lacey = []
        extrapolated_time = []

        lacey_grid_bin = GridBin(minCoords, maxCoords, *bins)

    for i in range(round(extrap_time/t_rnn)):
        pred_timestep = model.predict(last_seq)
        id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
        pred_timestep = np.hstack((pred_timestep, id_column))

        if stochastic_random == True:
            
            #fix particles to grid
            t1 = time()
            pred_timestep = fix_particle_coords(pred_timestep, drum_r, drum_w)
            t2 = time()

            #bin particles and apply velocity_std while tracking id
            # new_binned_particles = sr_grid_bin.get_binned_data(pred_timestep[:,:3], pred_timestep[:,3])
            new_positions, binned_indxs = sr_grid_bin.apply_random_velocity(pred_timestep[:,:3], velocity_stds, t_rnn)
            t3 = time()
            #print(f"Bining took {t3-t2:.2f}s")

            # np.unravel_index(14681, (sr_grid_bin.xBins, sr_grid_bin.yBins, sr_grid_bin.zBins))

            #print(f"RNG took {t4-t3:.2f}s")

            t5 = time()
            print(f"Total SR took {t5-t1:.2f}s. Binning took {t3-t2:.2f}s")


        if particle_loc_fix == True:
            new_positions = fix_particle_coords(new_positions, drum_r, drum_w)

        last_seq = last_seq[:, 1:, :]
        last_seq = np.concatenate((last_seq, new_positions[:, np.newaxis, :]), axis=1)

        if track_lacey == True:
            #break
            #calculate lacey index
            particle_types = np.array([id_dict.get(id, 0) for id in np.arange(1, new_positions.shape[0] + 1) ])
            binned_particle_types = lacey_grid_bin.get_particle_concentration(new_positions, particle_types)

            mass_1, mass_2, conc = unpack_mixing_results(lacey_grid_bin, binned_particle_types)
            Lacey_index = Lacey(mass_1, mass_2, conc, cut_off, len(new_positions))

            #append lacey data
            extrapolated_lacey.append(Lacey_index)
            time_i = start_t+(i+1)*t_rnn
            extrapolated_time.append(time_i)
            print(f"Lacey Index: {Lacey_index:.2f}")

        if save_plots == True:
            id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
            pred_timestep = np.hstack((pred_timestep, id_column))
            time_i = start_t+(i+1)*t_rnn
            plot_filename = rf"{plots_path}\{time_i:.2f}.png"
            plot_particles(pred_timestep, id_dict, True, time_i, plot_path=plot_filename)

        if show_plots and i % 5 == 0 and i != 0:
            id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
            pred_timestep = np.hstack((pred_timestep, id_column))
            time_i = start_t+(i+1)*t_rnn
            print(f"{time_i:.2f}s")
            plot_particles(pred_timestep, id_dict, True, time_i)



    #save lacey csv
    np.savetxt(rf"{plots_path}\_lacey.csv", np.column_stack((extrapolated_time, extrapolated_lacey)), delimiter=",")