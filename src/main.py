import matplotlib.pyplot as plt
import os
from extrapolation import extrapolation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

if __name__ == "__main__":

    #simulation parameters
    start_t = 1
    end_t = 10
    domain_x = (-0.06, 0.06)       
    domain_y = (-0.015, 0.015)
    domain_z = (-0.06, 0.06)
    num_bins = 2
    direction = "x"
    
    #get deck    
    simulation = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", '..', 'data', "rot_drum", "JKR_periodic_clean", "Rot_drum.dem"))
    sim_path = os.path.dirname(simulation)
    extrap = extrapolation(start_t, end_t, simulation, domain_x, domain_y, domain_z, num_bins, direction)

    #assing t1 and t2
    kinetic_energies, peak_times, peak_index, highlight_y = extrap.kin_energies(0.0002, 5)
    #get coordinates and split into subdomains
    particles_t1 = extrap.get_particle_coords(peak_index[1])
    particles_t2 = extrap.get_particle_coords(peak_index[2])
    slices_t1, slices_t2 = extrap.slice_particles(particles_t1, particles_t2)

    for slice in slices_t2:
        print(np.amin(slice[:,extrap.direction_dict[extrap.direction]]))

    #initialise position dictionary

    position_dictionary = {}

    for i in range(extrap.num_bins):
        particles_in_bin_t1 = slices_t1[i]
        particles_in_bin_t2 = slices_t2[i]

        if len(particles_in_bin_t1) != len(particles_in_bin_t2):

            print(f"before ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")
            if len(particles_in_bin_t1) > len(particles_in_bin_t2):
                    
                particles_in_bin_t2, slices_t2[i+1] = extrap.match_particle_numbers(particles_in_bin_t1, particles_in_bin_t2, slices_t2[i+1])

                print(f"after ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")
                
                distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)
                row_ind, col_ind = linear_sum_assignment(distance_matrix)

                pos_dict = extrap.position_dictionary(particles_in_bin_t1, particles_in_bin_t2, col_ind, particles_t2)

                for key in pos_dict:
                    if key in position_dictionary:
                        print(f'Key "{key}" exists in both dictionaries.')

                position_dictionary.update(pos_dict)
            
            elif len(particles_in_bin_t2) > len(particles_in_bin_t1):
                    
                particles_in_bin_t1, slices_t1[i+1] = extrap.match_particle_numbers(particles_in_bin_t2, particles_in_bin_t1, slices_t1[i+1])

                print(f"after ({i}): {len(particles_in_bin_t1), len(particles_in_bin_t2)}")

                distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)
                row_ind, col_ind = linear_sum_assignment(distance_matrix)

                pos_dict = extrap.position_dictionary(particles_in_bin_t1, particles_in_bin_t2, col_ind, particles_t2)
                position_dictionary.update(pos_dict)
        else:
            
            distance_matrix = cdist(particles_in_bin_t1, particles_in_bin_t2)
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
            pos_dict = extrap.position_dictionary(particles_in_bin_t1, particles_in_bin_t2, col_ind, particles_t2)
            position_dictionary.update(pos_dict)

        print(len(position_dictionary))
    
    dictionary_err = len(position_dictionary) - particles_t1.shape[0]
    if dictionary_err == 0:
        extrap.save_dict(sim_path, "peak23_4x_split", position_dictionary)
    else:
        raise ValueError(f"You fucked up. {abs(dictionary_err)} particles remain")
        
