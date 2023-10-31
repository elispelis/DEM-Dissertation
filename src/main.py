import matplotlib.pyplot as plt
import os
from LaceyClass import LaceyMixingAnalyzer
from extrapolation import extrapolation

if __name__ == "__main__":

    #simulation start and end time
    start_t = 1
    end_t = 45

    #get deck    
    simulation = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', "rot_drum", "JKR_periodic_clean", "Rot_drum.dem"))
    sim_path = os.path.dirname(simulation)
    extrap = extrapolation(start_t, end_t, simulation)

    #assign 0 or 1 to particles    
    id_dict = extrap.id_dictionary(extrap.sim_time[0])

    #assing t1 and t2
    kinetic_energies, peak_times, peak_index, highlight_y = extrap.kin_energies(0.0002, 5)
    #get coordinates and split into subdomains
    particles_t1 = extrap.get_particle_coords(peak_index[2])
    particles_t2 = extrap.get_particle_coords(peak_index[3])
    
    

    for i in range(extrap.num_bins):
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
