from edempy import Deck
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os

class extrapolation:
    def __init__(self, start_t, end_t, deck=None, domain_x=None, domain_y=None, domain_z=None, num_bins=None, direction=None):
        self.start_t = start_t
        self.end_t = end_t
        self.deck = None
        self.domain_x = domain_x 
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.num_bins = num_bins
        self.direction = str(direction)

        self.direction_dict = {
        "x": 0,
        "y": 1,
        "z": 2
        }

        if deck is not None:
            self.load_deck(deck)

        if start_t == 0:
            self.start= self.find_nearest(self.deck.timestepValues, deck.timestepValues[1])
        else:
            self.start = self.find_nearest(self.deck.timestepValues, start_t)

        #define time frame
        self.end=self.find_nearest(self.deck.timestepValues, end_t)
        self.sim_time = np.arange(self.start,self.end+1,1)
        self.sim_time=self.sim_time.astype(int)

    def load_deck(self, deck):
        self.deck = Deck(deck)

    def find_nearest(self, array, value):
            array=np.array(array)
            timestep = (np.abs(array-value)).argmin()
            return timestep

    def get_particle_coords(self, timestep):
        particle_n = 0
        x_coords = self.deck.timestep[timestep].particle[particle_n].getSphereXPositions()
        y_coords = self.deck.timestep[timestep].particle[particle_n].getSphereYPositions()
        z_coords = self.deck.timestep[timestep].particle[particle_n].getSphereZPositions()
        mass = self.deck.timestep[timestep].particle[particle_n].getMass()
        particle_ids = self.deck.timestep[timestep].particle[particle_n].getIds()

        return np.column_stack((x_coords, y_coords, z_coords, mass, particle_ids))

    def slice_particles(self, particle_locations_t1, particle_locations_t2):

        # Initialize empty lists for slices
        slices_t1 = []
        slices_t2 = []

        #bin_size = [(domain_x[1]-domain_x[0]) / num_bins, (domain_y[1]-domain_y[0]) / num_bins, (domain_z[1]-domain_z[0]) / num_bins]
        bin_size_x = (self.domain_x[1] - self.domain_x[0]) / self.num_bins
        bin_size_y = (self.domain_y[1] - self.domain_y[0]) / self.num_bins
        bin_size_z = (self.domain_z[1] - self.domain_z[0]) / self.num_bins

        # for i in range(self.num_bins):
        #         x_min = self.domain_x[0] + i * bin_size_x
        #         x_max = self.domain_x[0] + (i + 1) * bin_size_x
        #         slice_t1 = particle_locations_t1[(particle_locations_t1[:, self.direction_dict[self.direction]] >= x_min) & (particle_locations_t1[:, 0] < x_max)]
        #         slice_t2 = particle_locations_t2[(particle_locations_t2[:, self.direction_dict[self.direction]] >= x_min) & (particle_locations_t2[:, 0] < x_max)]
        #         slices_t1.append(slice_t1)
        #         slices_t2.append(slice_t2)

        # Split the domain into slices along the specified direction
        if self.direction == 'x':
            for i in range(self.num_bins):
                x_min = self.domain_x[0] + i * bin_size_x
                x_max = self.domain_x[0] + (i + 1) * bin_size_x
                slice_t1 = particle_locations_t1[(particle_locations_t1[:, 0] >= x_min) & (particle_locations_t1[:, 0] < x_max)]
                slice_t2 = particle_locations_t2[(particle_locations_t2[:, 0] >= x_min) & (particle_locations_t2[:, 0] < x_max)]
                slices_t1.append(slice_t1)
                slices_t2.append(slice_t2)
        elif self.direction == 'y':
            for j in range(self.num_bins):
                y_min = self.domain_y[0] + j * bin_size_y
                y_max = self.domain_y[0] + (j + 1) * bin_size_y
                slice_t1 = particle_locations_t1[(particle_locations_t1[:, 1] >= y_min) & (particle_locations_t1[:, 1] < y_max)]
                slice_t2 = particle_locations_t2[(particle_locations_t2[:, 1] >= y_min) & (particle_locations_t2[:, 1] < y_max)]
                slices_t1.append(slice_t1)
                slices_t2.append(slice_t2)
        elif self.direction == 'z':
            for k in range(self.num_bins):
                z_min = self.domain_z[0] + k * bin_size_z
                z_max = self.domain_z[0] + (k + 1) * bin_size_z
                slice_t1 = particle_locations_t1[(particle_locations_t1[:, 2] >= z_min) & (particle_locations_t1[:, 2] < z_max)]
                slice_t2 = particle_locations_t2[(particle_locations_t2[:, 2] >= z_min) & (particle_locations_t2[:, 2] < z_max)]
                slices_t1.append(slice_t1)
                slices_t2.append(slice_t2)

            slices_t1 = [np.array(slice) for slice in slices_t1]
            slices_t2 = [np.array(slice) for slice in slices_t2]

        return slices_t1, slices_t2
    
    def id_dictionary(self, init_time):
        #divide particles into two groups 
        particle_coords_start = self.get_particle_coords(init_time)
        particle_color = np.where(particle_coords_start[:,0] < 0, 1, 0)
        particle_coords_initial = np.column_stack((particle_coords_start, particle_color))
        id_dict = dict(particle_coords_initial[:,-2:])  # make the index a variable in the function

        return id_dict

    def plot_particles(self, particle_coords, id_dict, plot):
        if len(particle_coords[0,:])<6:
            id_color = np.array([id_dict.get(id,0) for id in particle_coords[:,-1]])
            particle_coords = np.column_stack((particle_coords, id_color))
            particle_coords = particle_coords[particle_coords[:,1].argsort()]

        if plot == True:
            #NEEDS CHANGING
            particle_plt_size = 10**4*(2/25.4)**2

            plt.scatter(particle_coords[:,0], particle_coords[:,2], s=particle_plt_size, edgecolors="k", c=particle_coords[:,-1], cmap="coolwarm")
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

        return particle_coords

    def kin_energies(self, threshold, point_distance):
        #measure high kinetic energies
        kinetic_energies = []
        n_particles = 1
        print("Analysing Kinetic Energies...")

        for m in self.sim_time:

            for n in range(n_particles):
                try:
                    m_kinEnergy = sum(self.deck.timestep[m].particle[n].getKineticEnergy())
                    kinetic_energies.append(m_kinEnergy)

                except:
                    continue

        #kinetic energy peaks
        kinetic_energies = np.array(kinetic_energies)*(-1)
        peak_index = find_peaks(kinetic_energies, height=-threshold, distance=point_distance)[0]+self.start #for all timesteps
        peak_times = np.array(self.deck.timestepValues)[peak_index]
        highlight_y = np.array(kinetic_energies)[np.searchsorted(self.deck.timestepValues[self.start:], peak_times)]

        return kinetic_energies, peak_times, peak_index, highlight_y
    
    def match_particle_numbers(self, higher_particle_n, lower_particle_n, particles_in_next_bin):
        num_extra_particles = len(higher_particle_n) - len(lower_particle_n)

        #sort particles
        particles_in_next_bin = particles_in_next_bin[particles_in_next_bin[:,self.direction_dict[self.direction]].argsort()]
        #get closest particles to edge 
        closest_particles_in_next_bin = particles_in_next_bin[:num_extra_particles]
        
        #append particles to balance 
        lower_particle_n = np.vstack((lower_particle_n, closest_particles_in_next_bin))
        
        #remove particles from consideration of the next bin
        particles_in_next_bin = particles_in_next_bin[num_extra_particles:]
        
        return lower_particle_n, particles_in_next_bin

    def hungarian_pairing(self, time1_particles, time2_particles):
        # Calculate the cost matrix based on Euclidean distance between coordinates
        cost_matrix = cdist(time1_particles[:, :3] , time2_particles[:, :3] )

        # Apply the Hungarian algorithm to find the optimal assignment
        hung_start_t = time.time()

        row_indices, col_indices = linear_sum_assignment(cost_matrix) ##row = time1, col = time2 -- eg time1_particles[index] == time2_particles[col_indices[index]], row indeces is from 0 upwards

        hung_end_t = time.time()
        hung_time = hung_end_t-hung_start_t

        return row_indices, col_indices, hung_time

    def position_dictionary(self, time1_particles, time2_particles, col_indices, global_time2_particles):
        predicted_time3 = np.empty_like(time1_particles)

        # Extrapolate positions from time 2 to time 3 for the matched particles
        for i, j in enumerate(col_indices):
            matching_id = time1_particles[i, -1]  # Get the ID from time 1
            matching_indices = np.where(global_time2_particles[:, -1] == matching_id)[0]  # Find matching IDs in time 2
            predicted_time3[i, :3] = global_time2_particles[matching_indices, :3] #New particle location
            predicted_time3[i, -1] = time2_particles[j, -1] #new particle ID

        #sort time2 and time 3 by ID number
        sorted_indices1 = np.argsort(time2_particles[:,-1])
        time2_particles = time2_particles[sorted_indices1]

        sorted_indices2 = np.argsort(predicted_time3[:,-1])
        predicted_time3 = predicted_time3[sorted_indices2]

        #create dictionary for tracking position change
        pos_dict = {}

        for key_row, value_row in zip(time2_particles[:,:3], predicted_time3[:,:3]):
            key = tuple(key_row)
            value = tuple(value_row)
            pos_dict[key] = value

        return pos_dict

    def extrapolate_particles(self, previous_particles, position_dictionary):
        extrapolated_particles = previous_particles

        for i, row in enumerate(extrapolated_particles):
            key = tuple(row[:3])
            if key in position_dictionary:
                extrapolated_particles[i,:3] = position_dictionary[key]
            else:
                raise KeyError("Previous particles are not found in dictionary")

        return extrapolated_particles

    def save_dict(self, dict_path, dict_name, pos_dict):
        name = str(dict_name) + "_dict.pkl"
        file_path = os.path.join(dict_path, name)
        
        with open(file_path, "wb") as file:
            pickle.dump(pos_dict, file)
        
        return print(file_path + " saved as dictionary")

    def import_dict(self, dict_path, dict_name):
        for file in os.listdir(dict_path):
            if file.startswith(dict_name):
                pos_dict_name = file
                with open(os.path.join(dict_path, pos_dict_name), "rb") as file:
                    pos_dict = pickle.load(file)
        return pos_dict

#extrap.plot_particles(extrap.extrapolate_particles(extrap.get_particle_coords()))



#get second timestep in simulation, if start_t == 0 (no particles exist at 0s)

# n_particles = deck.timestep[last_timestep].numTypes



# kinetic_energies, peak_times, peak_index, highlight_y = kin_energies(sim_time, start)

# plt.figure()
# plt.plot(deck.timestepValues[start:], kinetic_energies)
# plt.scatter(peak_times, highlight_y, c="red")




# plt.figure()
# plot_particles(particle_coords_start)



# #TO SHOW 3/10/23
# plot_particles(get_particle_coords(peak_index[3],0))

# pos_dict = import_dict("peak23")
# plot_particles(extrapolate_particles(get_particle_coords(peak_index[2],0), pos_dict))


##PLOT PEAKS
# for x in peak_index: 
#     plt.figure()
#     particle_coords = get_particle_coords(x, 0)
#     # id_color = np.array([id_dict.get(id,0) for id in particle_coords[:,-1]])
#     # particle_coords = np.column_stack((particle_coords, id_color))
#     plot_particles(particle_coords)

# plt.show



# pos_dict = position_dictionary(time1_particles=time1_particles, time2_particles=time2_particles, col_indices=col_indices)

# #extrapolation loop
# time2_particles = get_particle_coords(peak_index[1],0)
# predicted_old = extrapolate_particles(time2_particles, pos_dict)

# for i in range(20):
#     plt.figure()
#     predicted_new = extrapolate_particles(predicted_old, pos_dict)
#     plot_particles(predicted_new)
#     predicted_old = predicted_new


#PLOTTING ALL PEAK INDEXES
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# plots_y = len(peak_index) + 1


#PLOTTING ALL PEAK INDEXES
# def plot_particles(particle_coords, subplot_row, total_rows):
#     position = subplot_row * 2 + 1
    
#     id_color = np.array([id_dict.get(id, 0) for id in particle_coords[:, -1]])
#     particle_coords = np.column_stack((particle_coords, id_color))
#     particle_coords = particle_coords[particle_coords[:, 1].argsort()]

#     # Create a subplot in the first column at the specified row
#     ax = plt.subplot(total_rows, 2, position)

#     # Scatter plot as per your existing code
#     plt.scatter(particle_coords[:, 0], particle_coords[:, 2], s=particle_plt_size, edgecolors="k", c=particle_coords[:, -1], cmap="coolwarm")

#     plt.title(f'Subplot {subplot_row + 1} (Left Column)')

#     ax.set_aspect("equal")

# # Example usage:
# fig = plt.figure(figsize=(50, 50))
# # Define a gridspec to control subplot sizes
# gs = gridspec.GridSpec(plots_y, 2, width_ratios=[1, 1])

# # Iterate through the rows, calling the function for each row
# for row in range(plots_y):
#     # Pass the particle_coords, row, and total_rows to the function
#     if row == 0:
#         plot_particles(get_particle_coords(start, 0), row, plots_y)
#     else:     
#         plot_particles(get_particle_coords(peak_index[row - 1], 0), row, plots_y)

# plt.tight_layout()
# plt.show()