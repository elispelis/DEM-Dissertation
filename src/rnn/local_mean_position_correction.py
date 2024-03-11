import pandas as pd
import numpy as np

data_paths = ["../model/3_4_0.05s.csv", "../model/3_4_0.01s.csv", "../model/4_6_0.05s.csv", "../../model/3_7_0.02s.csv",  "../../model/Rot_drum_400k_3_5_0.05s.csv", "../../model/Rot_drum_400k_3_5_0.03s.csv", "../../model/Rot_drum_400k_3_6.5_0.05s.csv", "../../model/Rot_drum_400k_3_4.4_0.02s.csv"]
data_path = data_paths[-1]

df = pd.read_csv(data_path, index_col=0)

num_features = 3
num_timesteps = df.shape[1] // num_features
num_particles = df.shape[0]
drum_r = 0.07 #m
drum_w = 0.025 #m

local_means = df.values.reshape(num_particles, num_timesteps, num_features)

adj_count = 0

for i in range(num_timesteps):
    local_mean = local_means[:,i,:]
    distances = np.sqrt((local_mean[:, 0])**2+(local_mean[:, 2])**2)
    violating_particles = np.where(distances>drum_r)

    adj_count += len(violating_particles)

    for particle_idx in violating_particles:
        # Normalize the position vector
        norm_factor = drum_r / distances[particle_idx]
        local_mean[particle_idx, 0] *= norm_factor
        local_mean[particle_idx, 2] *= norm_factor
    
    violating_particles = np.where(abs(local_mean[:,1])>drum_w)

    adj_count += len(violating_particles)

    for particle_idx_w in violating_particles:
            sign_w = np.sign(local_mean[particle_idx_w, 1])
            local_mean[particle_idx_w, 1] = sign_w * drum_w

    local_means[:,i,:] = local_mean

reshaped_array = local_means.reshape(num_particles, num_timesteps * num_features)

# Create a new DataFrame with the reshaped array
df.iloc[:,:] = reshaped_array
df.to_csv(f"{data_path[:-4]}_adj.csv")
print(f"Fixed {adj_count} particles")