import numpy as np

# Define your domain boundaries, the number of bins, and their dimensions
domain_x = 1
domain_y = 1
domain_z = 1
num_bins = 10
direction = "y"


np.random.seed(0)

# Generate random particle locations within the 3D domain
num_particles = 10000
particle_locations_t1 = np.random.rand(num_particles, 3) * np.array([domain_x, domain_y, domain_z])
particle_locations_t2 = np.random.rand(num_particles, 3) * np.array([domain_x, domain_y, domain_z])

def slice_particles(domain_x, domain_y, domain_z, num_bins, direction, particle_locations_t1, particle_locations_t2):

    # Initialize empty lists for slices
    slices_t1 = []
    slices_t2 = []

    bin_size = [domain_x / num_bins, domain_y / num_bins, domain_z / num_bins]

    # Split the domain into slices along the specified direction
    if direction == 'x':
            slices_t1 = [particle_locations_t1[(particle_locations_t1[:, 0] >= i * bin_size[0]) & (particle_locations_t1[:, 0] < (i + 1) * bin_size[0])] for i in range(num_bins)]
            slices_t2 = [particle_locations_t2[(particle_locations_t2[:, 0] >= i * bin_size[0]) & (particle_locations_t2[:, 0] < (i + 1) * bin_size[0])] for i in range(num_bins)]
    elif direction == "y":
            slices_t1 = [particle_locations_t1[(particle_locations_t1[:, 1] >= j * bin_size[1]) & (particle_locations_t1[:, 1] < (j + 1) * bin_size[1])] for j in range(num_bins)]
            slices_t2 = [particle_locations_t2[(particle_locations_t2[:, 1] >= j * bin_size[1]) & (particle_locations_t2[:, 1] < (j + 1) * bin_size[1])] for j in range(num_bins)]
    elif direction == "z":
            slices_t2 = [particle_locations_t2[(particle_locations_t2[:, 2] >= k * bin_size[2]) & (particle_locations_t2[:, 2] < (k + 1) * bin_size[2])] for k in range(num_bins)]
            slices_t1 = [particle_locations_t1[(particle_locations_t1[:, 2] >= k * bin_size[2]) & (particle_locations_t1[:, 2] < (k + 1) * bin_size[2])] for k in range(num_bins)]

    slices_t1 = [np.array(slice) for slice in slices_t1]
    slices_t2 = [np.array(slice) for slice in slices_t2]

    return slices_t1, slices_t2

slices_t1, slices_t2 = slice_particles(domain_x, domain_y, domain_z, num_bins, direction, particle_locations_t1, particle_locations_t2)

for i in range(num_bins):
    if direction == "x":
        print(np.amax(slices_t2[i][:,0]))
    elif direction =="y":
        print(np.amax(slices_t2[i][:,1]))
    elif direction == "z":
        print(np.amax(slices_t2[i][:,2]))

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
        
        elif len(particles_in_bin_t2) > len(particles_in_bin_t1):
                
            particles_in_bin_t1, slices_t1[i+1] = match_particle_numbers(particles_in_bin_t2, particles_in_bin_t1, direction, slices_t1[i+1])

            print(f"after ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")
        