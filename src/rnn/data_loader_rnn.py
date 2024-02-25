import os
import csv
import pandas as pd
from edempy import Deck
import numpy as np

class RNNLoader:
    def __init__(self, start_t, end_t, deck):
        self.start_t = start_t
        self.end_t = end_t
        self.deck = Deck(deck)
        # self.domain_x = domain_x        
        # self.domain_y = domain_y
        # self.domain_z = domain_z
        # self.num_bins = num_bins
        # self.direction = str(direction)

        self.direction_dict = {
        "x": 0,
        "y": 1,
        "z": 2
        }

        if start_t == 0:
            self.start= self.find_nearest(self.deck.timestepValues, deck.timestepValues[1])
        else:
            self.start = self.find_nearest(self.deck.timestepValues, start_t)

        #define time frame
        self.end=self.find_nearest(self.deck.timestepValues, end_t)
        self.sim_time = np.arange(self.start,self.end+1,1)
        self.sim_time=self.sim_time.astype(int)

    def find_nearest(self, array, value):
        array=np.array(array)
        timestep = (np.abs(array-value)).argmin()
        return timestep

    def get_particle_data(self, timestep):
        particle_n = 0  # change for more than 1 particle type
        x_coords = self.deck.timestep[timestep].particle[particle_n].getSphereXPositions()
        y_coords = self.deck.timestep[timestep].particle[particle_n].getSphereYPositions()
        z_coords = self.deck.timestep[timestep].particle[particle_n].getSphereZPositions()

        x_vel = self.deck.timestep[timestep].particle[particle_n].getXVelocities()
        y_vel = self.deck.timestep[timestep].particle[particle_n].getYVelocities()
        z_vel = self.deck.timestep[timestep].particle[particle_n].getZVelocities()

        mass = self.deck.timestep[timestep].particle[particle_n].getMass()
        particle_ids = self.deck.timestep[timestep].particle[particle_n].getIds()

        return np.column_stack((x_coords, y_coords, z_coords, x_vel, y_vel, z_vel, mass, particle_ids))
    
    def read_p4_data(data_path):
        # Iterate through all files in the specified folder
        for filename in os.listdir(data_path):
            if filename.endswith("00100001.p4p"):
                # Build the full path to the CSV file
                file_path = os.path.join(data_path, filename)
                
                # Read the CSV file into a DataFrame, skipping the first two rows
                df = pd.read_csv(file_path, sep=r'\s+', header=2)
        
        return df
    
    def local_mean_position(self, delta_t):

        num_particles = self.deck.timestep[rnn.start].particle[0].getNumParticles()  # change for more than 1 particle type
        df = pd.DataFrame(index=np.arange(1,num_particles+1,1))

        for timestep in np.arange(self.start_t, self.end_t, delta_t):
            timestep_index = self.find_nearest(self.deck.timestepValues, timestep)

            particle_data = self.get_particle_data(timestep_index)
            coords = particle_data[:,:3]
            vels = particle_data[:, 3:6]
            p_id = particle_data[:, -1]

            local_mean_pos = coords - vels * delta_t

            previous_t = timestep - delta_t
            
            temp_df = pd.DataFrame(local_mean_pos, columns=[f"X_{previous_t:.2f}", f"Y_{previous_t:.2f}", f"Z_{previous_t:.2f}"], index=p_id)

            df = pd.merge(df, temp_df, left_index=True, right_index=True, how='outer')
        
        return df



if __name__ == "__main__":
        
    sim_names = ["Rot_drum_mono.dem", "Rot_drum_binary_mixed.dem", "Rot_drum_400k.dem"]
    sim_name = sim_names[0]
    sim_path =rf"V:\GrNN_EDEM-Sims\{sim_name}"

    start_t = 3
    end_t = 7
    rnn = RNNLoader(start_t,end_t,sim_path)

    delta_t_rnn = 0.02

    print("Generating DataFrame...")
    rnn_df = rnn.local_mean_position(delta_t_rnn)
    rnn_df.to_csv(f"../../model/{sim_name[:-4]}_{start_t}_{end_t}_{delta_t_rnn}s.csv")




# for filename in sorted(os.listdir(default_data_path)):
#     if filename.endswith(".p4p"):
#         # Build the full path to the CSV file
#         file_path = os.path.join(default_data_path, filename)
        
#         print(filename)



# for file in files: 
#     get timestep
#     get coords
#     get velocites
#     x_minus_delta_t = coords - velocities*timestep

#     df  