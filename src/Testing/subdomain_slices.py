import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Define your domain boundaries, the number of bins, and their dimensions
domain_x = (-1, 1)
domain_y = (-1, 1)
domain_z = (-1, 1)
num_bins = 10
direction = "y"


np.random.seed(0)

# Generate random particle locations within the 3D domain
num_particles = 10000
particle_locations_t1 = np.random.uniform(low=[domain_x[0], domain_y[0], domain_z[0]], high=[domain_x[1], domain_y[1], domain_z[1]], size=(num_particles, 3))
particle_locations_t2 = np.random.uniform(low=[domain_x[0], domain_y[0], domain_z[0]], high=[domain_x[1], domain_y[1], domain_z[1]], size=(num_particles, 3))

optimal_pairings = []

optimal_pairings = []

def slice_particles(domain_x, domain_y, domain_z, num_bins, direction, particle_locations_t1, particle_locations_t2):

    # Initialize empty lists for slices
    slices_t1 = []
    slices_t2 = []

    #bin_size = [(domain_x[1]-domain_x[0]) / num_bins, (domain_y[1]-domain_y[0]) / num_bins, (domain_z[1]-domain_z[0]) / num_bins]
    bin_size_x = (domain_x[1] - domain_x[0]) / num_bins
    bin_size_y = (domain_y[1] - domain_y[0]) / num_bins
    bin_size_z = (domain_z[1] - domain_z[0]) / num_bins

    # Split the domain into slices along the specified direction
    if direction == 'x':
        for i in range(num_bins):
            x_min = domain_x[0] + i * bin_size_x
            x_max = domain_x[0] + (i + 1) * bin_size_x
            slice_t1 = particle_locations_t1[(particle_locations_t1[:, 0] >= x_min) & (particle_locations_t1[:, 0] < x_max)]
            slice_t2 = particle_locations_t2[(particle_locations_t2[:, 0] >= x_min) & (particle_locations_t2[:, 0] < x_max)]
            slices_t1.append(slice_t1)
            slices_t2.append(slice_t2)
    elif direction == 'y':
        for j in range(num_bins):
            y_min = domain_y[0] + j * bin_size_y
            y_max = domain_y[0] + (j + 1) * bin_size_y
            slice_t1 = particle_locations_t1[(particle_locations_t1[:, 1] >= y_min) & (particle_locations_t1[:, 1] < y_max)]
            slice_t2 = particle_locations_t2[(particle_locations_t2[:, 1] >= y_min) & (particle_locations_t2[:, 1] < y_max)]
            slices_t1.append(slice_t1)
            slices_t2.append(slice_t2)
    elif direction == 'z':
        for k in range(num_bins):
            z_min = domain_z[0] + k * bin_size_z
            z_max = domain_z[0] + (k + 1) * bin_size_z
            slice_t1 = particle_locations_t1[(particle_locations_t1[:, 2] >= z_min) & (particle_locations_t1[:, 2] < z_max)]
            slice_t2 = particle_locations_t2[(particle_locations_t2[:, 2] >= z_min) & (particle_locations_t2[:, 2] < z_max)]
            slices_t1.append(slice_t1)
            slices_t2.append(slice_t2)

    slices_t1 = [np.array(slice) for slice in slices_t1]
    slices_t2 = [np.array(slice) for slice in slices_t2]

    return slices_t1, slices_t2

slices_t1, slices_t2 = slice_particles(domain_x, domain_y, domain_z, num_bins, direction, particle_locations_t1, particle_locations_t2)

# for i in range(num_bins):
#     if direction == "x":
#         print(np.amax(slices_t2[i][:,0]))
#     elif direction =="y":
#         print(np.amax(slices_t2[i][:,1]))
#     elif direction == "z":
#         print(np.amax(slices_t2[i][:,2]))

def match_particle_numbers(higher_particle_n, lower_particle_n, direction, particles_in_next_bin):
    num_extra_particles = len(higher_particle_n) - len(lower_particle_n)

    direction_dict = {
     "x": 0,
     "y": 1,
     "z": 2
    }

    #get closest particles to edge 
    closest_particles_in_next_bin = particles_in_next_bin[particles_in_next_bin[:,direction_dict[direction]].argsort()][:num_extra_particles]
    
    #append particles to balance 
    lower_particle_n = np.vstack((lower_particle_n, closest_particles_in_next_bin))
    
    #remove particles from consideration of the next bin
    particles_in_next_bin = particles_in_next_bin[num_extra_particles:]
    
    return lower_particle_n, particles_in_next_bin


for i in range(num_bins):
    particles_in_bin_t1 = slices_t1[i]
    particles_in_bin_t2 = slices_t2[i]

    if len(particles_in_bin_t1) != len(particles_in_bin_t2):

        print(f"before ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")
        if len(particles_in_bin_t1) > len(particles_in_bin_t2):
                
            particles_in_bin_t2, slices_t2[i+1] = match_particle_numbers(particles_in_bin_t1, particles_in_bin_t2, direction, slices_t2[i+1])

            print(f"after ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")
            
            distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)
            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            # Store the optimal pairings for the current bin
            for r, c in zip(row_ind, col_ind):
                particle_t1 = particles_in_bin_t1[r]
                particle_t2 = particles_in_bin_t2[c]
                optimal_pairings.append((particle_t1, particle_t2))


        
        elif len(particles_in_bin_t2) > len(particles_in_bin_t1):
                
            particles_in_bin_t1, slices_t1[i+1] = match_particle_numbers(particles_in_bin_t2, particles_in_bin_t1, direction, slices_t1[i+1])

            print(f"after ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")

            distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)
            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            # Store the optimal pairings for the current bin
            for r, c in zip(row_ind, col_ind):
                particle_t1 = particles_in_bin_t1[r]
                particle_t2 = particles_in_bin_t2[c]
                optimal_pairings.append((particle_t1, particle_t2))

    else:
        
        distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # Store the optimal pairings for the current bin
        for r, c in zip(row_ind, col_ind):
            particle_t1 = particles_in_bin_t1[r]
            particle_t2 = particles_in_bin_t2[c]
            optimal_pairings.append((particle_t1, particle_t2))
        

print(np.array(optimal_pairings).shape)
        