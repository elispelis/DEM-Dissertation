import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

#test

def keep_Nc_smallest(matrix, Nc):
    # Create a copy of the matrix
    result = matrix.copy()
    
    # Find the Nc smallest values in each row
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        indices = np.argpartition(row, Nc)[:Nc]
        result[i, indices] = row[indices]
        result[i, np.setdiff1d(np.arange(matrix.shape[1]), indices)] = 0
    
    return result

# Define your domain boundaries, the number of bins, and their dimensions
domain_width = 1
domain_height = 1
num_bins_x = 5
num_bins_y = 5
bin_width = domain_width / num_bins_x
bin_height = domain_height / num_bins_y

np.random.seed(0)
dense = 1 

# Generate random particle locations at time t1 and t2 within the domain
num_particles = 10000
Nc = 30
particle_locations_t1 = np.random.rand(num_particles, 2) * np.array([domain_width, domain_height])
particle_locations_t2 = np.random.rand(num_particles, 2) * np.array([domain_width, domain_height])

# Initialize an empty list to store optimal pairings
optimal_pairings = []

# Iterate through bins
for bin_x in range(num_bins_x):
    for bin_y in range(num_bins_y):
        # Define the boundaries of the current bin
        bin_x_min = bin_x * bin_width
        bin_x_max = (bin_x + 1) * bin_width
        bin_y_min = bin_y * bin_height
        bin_y_max = (bin_y + 1) * bin_height

        # Filter particles that fall within the current bin for both times
        particles_in_bin_t1 = [p for p in particle_locations_t1 if bin_x_min <= p[0] < bin_x_max and bin_y_min <= p[1] < bin_y_max]
        particles_in_bin_t2 = [p for p in particle_locations_t2 if bin_x_min <= p[0] < bin_x_max and bin_y_min <= p[1] < bin_y_max]

        print(f"Before: {len(particles_in_bin_t1)},{len(particles_in_bin_t2)}")

        if len(particles_in_bin_t1) != len(particles_in_bin_t2):
            if len(particles_in_bin_t1) > len(particles_in_bin_t2):
                # Calculate the number of extra particles
                num_extra_particles = len(particles_in_bin_t1) - len(particles_in_bin_t2)

                # Find the particles in the next bin (bin_x + 1) that are closest to the edge
                next_bin_x_min = bin_x_min + bin_width
                next_bin_x_max = bin_x_max + bin_width
                next_bin_y_min = bin_y_min + bin_height
                next_bin_y_max = bin_y_max + bin_height

                # Filter particles that fall within the next bin
                particles_in_next_bin_t2 = [p for p in particle_locations_t2 if bin_x_min <= p[0] < bin_x_max and bin_y_min <= p[1] < bin_y_max]

                # Sort the next bin particles by distance to the edge
                next_bin_particles_sorted = sorted(particles_in_next_bin_t2, key=lambda p: abs(p[1] - next_bin_y_min))

                # Take the first 'num_extra_particles' particles from the next bin
                selected_particles_from_next_bin = next_bin_particles_sorted[:num_extra_particles]

                # Add these selected particles to the current bin
                particles_in_bin_t2.extend(selected_particles_from_next_bin)

                # Remove these particles from the global domain
                particle_locations_t2 = [p for p in particle_locations_t2 if p not in selected_particles_from_next_bin]

        print(f"After: {len(particles_in_bin_t1)},{len(particles_in_bin_t2)}")
        
        if len(particles_in_bin_t1) != len(particles_in_bin_t2):
            raise ValueError("Try again")



        # Calculate pairwise distances between particles in the current bin using cdist
        distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)

        if dense == True: 
            for row in distance_matrix:
                # Find the Nc'th smallest element in the sub-array
                Nc_smallest = np.partition(row, Nc)[Nc]
                
                # Make Nc_smallest and smaller elements negative
                #row[row <= Nc_smallest] *= -1
                
                # Set larger elements to 0
                row[row > Nc_smallest] = 0

            # Solve the linear sum assignment problem using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(distance_matrix)



        # Store the optimal pairings for the current bin
        for r, c in zip(row_ind, col_ind):
            particle_t1 = particles_in_bin_t1[r]
            particle_t2 = particles_in_bin_t2[c]
            optimal_pairings.append((particle_t1, particle_t2))

# Now, 'optimal_pairings' contains the optimal pairings of particles within each bin
print(np.array(optimal_pairings).shape)



        # else: 
        #     # Create a sparse matrix from the distance_matrix
        #     # Keep the Nc smallest elements in each row and make others 0
        #     sparse_distance_matrix = sp.csr_matrix(keep_Nc_smallest(distance_matrix, Nc))

        #     # Find the minimum weight matching using min_weight_full_bipartite_matching
        #     row_ind, col_ind = min_weight_full_bipartite_matching(sparse_distance_matrix)