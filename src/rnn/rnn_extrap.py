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
import math


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

        ax.add_patch(plt.Circle((0,0), 0.07, color="lightblue", zorder = 0))
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

def get_daor(particles, grid_y, grid_x, bin_size):
    delta=np.zeros(len(grid_y))
    index_nonzero=np.zeros(len(grid_x))
    SurfaceZ=np.zeros(shape=(len(grid_y),len(grid_x)))
    SurfaceY=np.zeros(shape=(len(grid_y),len(grid_x)))
    SurfaceX=np.zeros(shape=(len(grid_y),len(grid_x)))
    Coord=particles[:,:3]

    for i in range(len(grid_y)):
    #Find surface particles
        for j in range (len(grid_x)):
            index_coord=np.where((Coord[:,0]>(grid_x[j]-bin_size[0]/2)) & (Coord[:,0]<(grid_x[j]+bin_size[0]/2)) & (Coord[:,1]>(grid_y[i]-bin_size[1]/2)) & (Coord[:,1]<(grid_y[i]+bin_size[1]/2)))
            surf=Coord[index_coord]
            #Index zero values and get surface particles
            if surf.shape[0]>0:
                Max=np.argmax(surf[:,2])
                SurfaceX[i][j]=surf[Max,0]
                SurfaceY[i][j]=surf[Max,1]
                SurfaceZ[i][j]=surf[Max,2]
                index_nonzero[j]=j
            else:
                index_nonzero[j]=-1
        #Linear fit to surface particles
        fit=np.polyfit(grid_x[index_nonzero!=-1],SurfaceZ[i][index_nonzero!=-1],1)
        #Calculating angle of repose and statistics
        delta[i]=math.atan(abs(fit[0]))*180/math.pi

    
    delta_mean = np.average(delta)
    delta_std = np.std(delta)
    delta_cov = delta_std/delta_mean*100

    return delta_mean, delta_std, delta_cov

if __name__ == "__main__":
    sim_names = ["Rot_drum_mono", "Rot_drum_binary_mixed", "Rot_drum_400k"]
    sim_name = sim_names[-1]
    sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}.dem"
    id_dict_path = rf"V:\GrNN_EDEM-Sims\{sim_name}_data\Export_Data"
    model_paths = ["../../model/model_sl10_tr144.h5", "../../model/model_sl50_tr80.h5", "../../model/model_sl15_tr36.h5", "../../model/model_sl15_tr36_adj.h5", 
                   "../../model/model_sl25_tr90_adj.h5", "../../model/model_sl25_tr180_adj.h5", "../../model/model_sl15_tr36_adj_big.h5",
                    "../../model/model_sl30_tr36_adj_big.h5" , "../../model/400k_sl25_tr60_adj.h5", "../../model/3_6.5_model_sl15_tr63_adj.h5", 
                    "../../model/model_sl30_tr60_adj_64batch_0.03s.h5", "../../model/3_4.4_0.02s_model_sl38_tr64_adj.h5","../../model/3_4.4_0.02s_model_sl25_tr64_adj.h5",
                    "../../model/model_sl17_tr60_3_5_0.03s_adj_128batch_30epoch.h5", "../../model/model_sl15_tr63_3_6.5_30epoch_128batch_adj.h5", "../../model/model_sl15_tr36_3_5_0.05s_20ep_32_batch_adj.h5",
                    "../../model/model_sl25_tr60_3_5_0.03s_20epoch_128batch_adj.h5", "../../model/model_sl50_tr60_3_5_0.03s_30ep_128batch_adj.h5", "../../model/model_sl25_tr60_3_5_0.03s_20ep_64batch_adj.h5"]
    
    data_paths = ["../../model/3_4_0.05s.csv", "../../model/3_4_0.01s.csv", "../../model/4_6_0.05s.csv", "../../model/4_6_0.05s_adj.csv", "../../model/3_4_0.01s.csv", 
                  "../../model/3_7_0.02s_adj.csv", "../../model/Rot_drum_400k_3_5_0.05s_adj.csv", "../../model/Rot_drum_400k_3_5_0.03s_adj.csv", "../../model/Rot_drum_400k_3_6.5_0.05s_adj.csv", "../../model/Rot_drum_400k_3_4.4_0.02s_adj.csv"] 

    model_path = model_paths[-1]
    data_path = data_paths[-3]

    #load id dictionary, model and starting series
    id_dict = import_dict(id_dict_path, "id_dict")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')

    df = pd.read_csv(data_path, index_col=0)
    num_features = 3
    num_timesteps = df.shape[1] // num_features
    num_particles = df.shape[0]
    seq_length = 25

    last_seq = df.iloc[:, (num_timesteps-seq_length)*num_features:]
    last_seq = last_seq.values.reshape(-1, seq_length, num_features)

    #extrapolation settings
    particle_loc_fix = False
    stochastic_random = True
    track_lacey = True
    track_DAoR = False
    save_plots = True
    show_plots = True
    save_coords = True

    start_t = 4.95
    t_rnn = 0.03
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

        velocity_std_path = rf"{sim_path[:-4]}_data\Export_Data\{bins[0]}_{bins[1]}_{bins[2]}_{Ng}.npy"

        #b_coords, div_size = lacey.grid()
        sr_grid_bin = GridBin(minCoords, maxCoords, *bins)
        #velocity_stds = np.genfromtxt(velocity_std_path, delimiter=",")
        velocity_stds = np.load(velocity_std_path)

    if save_plots == True:
        show_plots = False
        plots_path = rf"{sim_path[:-4]}_data\Export_Data\RNNSR_plots\{bins[0]}_{bins[1]}_{bins[2]}_sl{seq_length}_3_5_{t_rnn}s_plots_20ep_64batch"
        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(rf"{plots_path}\timestep_data", exist_ok=True)

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

        lacey = LaceyMixingAnalyzer(minCoords, maxCoords, bins)

        b_coords_lacey, div_size_lacey = lacey.grid()
        extrapolated_lacey = []
        extrapolated_time = []

        lacey_grid_bin = GridBin(minCoords, maxCoords, *bins)

    if track_DAoR == True:
        daor_settings = f"{sim_path[:-4]}_data\Export_Data\Dynamic_angle_of_repose_analyst_settings.txt"

        with open(daor_settings, 'r') as file:
            preferences=file.readlines()
            sim_end=float(preferences[3])
            domain=np.array(preferences[5].split(','))
            domain=domain.astype('float64')
            bin_size=np.array(preferences[7].split(','))
            bin_size=bin_size.astype('float64')
            file.close()

        grid_x=np.linspace(domain[0],domain[1],int((domain[1]-domain[0])/bin_size[0]))
        grid_y=np.linspace(domain[2],domain[3],int((domain[3]-domain[2])/bin_size[1]))

        delta_means = []
        delta_stds = []
        delta_covs = []

    profiles_fixed = []
    sides_fixed = []

    #MAIN LOOP
    for i in range(round(extrap_time/t_rnn)):
        pred_timestep = model.predict(last_seq)
        id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
        pred_timestep = np.hstack((pred_timestep, id_column))

        profile_i_fixed = 0
        side_i_fixed = 0
        
        if stochastic_random == True:
            
            #fix particles to grid
            t1 = time()
            pred_timestep, profile_fixed, side_fixed = fix_particle_coords(pred_timestep, drum_r, drum_w)
            t2 = time()

            profile_i_fixed += profile_fixed
            side_i_fixed += side_fixed

            #bin particles and apply velocity_std while tracking id
            # new_binned_particles = sr_grid_bin.get_binned_data(pred_timestep[:,:3], pred_timestep[:,3])
            pred_timestep, binned_indxs = sr_grid_bin.apply_random_velocity(pred_timestep[:,:3], velocity_stds, t_rnn)
            t3 = time()
            #print(f"Bining took {t3-t2:.2f}s")
            # np.unravel_index(14681, (sr_grid_bin.xBins, sr_grid_bin.yBins, sr_grid_bin.zBins))

            #print(f"RNG took {t4-t3:.2f}s")

            t5 = time()
            print(f"Total SR took {t5-t1:.2f}s. Binning took {t3-t2:.2f}s")


        if particle_loc_fix == True:
            pred_timestep, profile_fixed, side_fixed = fix_particle_coords(pred_timestep, drum_r, drum_w)

            profile_i_fixed += profile_fixed
            side_i_fixed += side_fixed

        profiles_fixed.append(profile_i_fixed)
        sides_fixed.append(side_i_fixed)

        last_seq = last_seq[:, 1:, :]
        last_seq = np.concatenate((last_seq, pred_timestep[:, np.newaxis, :]), axis=1)

        if track_lacey == True:
            #break
            #calculate lacey index
            particle_types = np.array([id_dict.get(id, 0) for id in np.arange(1, pred_timestep.shape[0] + 1) ])
            binned_particle_types = lacey_grid_bin.get_particle_concentration(pred_timestep, particle_types)

            mass_1, mass_2, conc = unpack_mixing_results(lacey_grid_bin, binned_particle_types)
            Lacey_index = Lacey(mass_1, mass_2, conc, cut_off, len(pred_timestep))

            #append lacey data
            extrapolated_lacey.append(Lacey_index)
            time_i = start_t+(i+1)*t_rnn
            extrapolated_time.append(time_i)
            print(f"Lacey Index: {Lacey_index:.2f}")

        if track_DAoR == True:
            delta_mean, delta_std, delta_cov = get_daor(pred_timestep, grid_y, grid_x, bin_size)
            print(f"DAoR: {delta_mean:.2f}")
            delta_means.append(delta_mean)
            delta_stds.append(delta_std)
            delta_covs.append(delta_cov)

        if save_plots == True:
            id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
            pred_timestep = np.hstack((pred_timestep, id_column))
            time_i = start_t+(i+1)*t_rnn
            plot_filename = rf"{plots_path}\{time_i:.2f}.png"
            plot_particles(pred_timestep, id_dict, True, time_i, plot_path=plot_filename)
        
        if save_coords == True:
            #np.savetxt(rf"{plots_path}\timestep_data\{time_i:.2f}.csv", pred_timestep, delimiter=",")
            
            with open(rf"{plots_path}\timestep_data\{time_i:.2f}.npy", 'wb') as f:
                np.save(f, pred_timestep)

        if show_plots and i % 5 == 0 and i != 0:
            id_column = np.arange(1, pred_timestep.shape[0] + 1).reshape(-1, 1)
            pred_timestep = np.hstack((pred_timestep, id_column))
            time_i = start_t+(i+1)*t_rnn
            print(f"{time_i:.2f}s")
            plot_particles(pred_timestep, id_dict, True, time_i)



    #save lacey csv
    np.savetxt(rf"{plots_path}\_lacey.csv", np.column_stack((extrapolated_time, extrapolated_lacey)), delimiter=",")
    np.savetxt(rf"{plots_path}\_fixed_particles.csv", np.column_stack((extrapolated_time, profiles_fixed, sides_fixed)), delimiter=",")
    np.savetxt(rf"{plots_path}\_DAoR.csv", np.column_stack((extrapolated_time, delta_means, delta_stds, delta_covs)), delimiter=",")