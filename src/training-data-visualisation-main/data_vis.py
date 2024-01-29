"""
Plotttting Ascii GNS Data
"""

#%% Imports

import pathlib
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import numpy as np
import pyvista as pv
# from pyvistaqt import BackgroundPlotter

from helpers import read_metadata, get_p4_files_list, read_gns_file, Cylinder
from plotting_module import create_particle_mesh, plot_simulation_timestep, animate_particles

#%%  Functions

def calculate_wall_distance(timestep_data, drum):
    dist_end_1 = np.dot(timestep_data.position - drum.point1, drum.vector_norm)
    dist_end_2 = np.dot(drum.point2 - timestep_data.position, drum.vector_norm)
    radial_dist_to_wall = np.linalg.norm(np.cross(timestep_data.position - drum.point1, drum.vector), axis=1) / np.linalg.norm(drum.vector)

    all_dists = np.stack([dist_end_1, dist_end_2, radial_dist_to_wall]).T

    dist_to_wall = np.min(all_dists, 1)

    return dist_to_wall


#%% Calculate wall distances
datapath = r'V:\GrNN_EDEM-Sims\Rot_drum_bi_segregated_data\Export_Data'
metadata = read_metadata(datapath)
particle_files_list = get_p4_files_list(datapath)

timestep_data = read_gns_file(gns_particle_file=particle_files_list[1000])

drum = Cylinder(metadata['geometry']['axis_start'], 
                metadata['geometry']['axis_end'], 
                metadata['geometry']['radius'])

wall_distances = calculate_wall_distance(timestep_data, drum)


#%% 3D Particle Plotting
timestep_data = read_gns_file(gns_particle_file=particle_files_list[566])
particle_mesh_initial = create_particle_mesh(timestep_data)

cylinder = pv.Cylinder(center=drum.point1+drum.vector/2, 
direction=drum.vector_norm, 
radius=drum.radius, 
height=drum.cylinder_height)


geoms_to_plot = {'drum': {'geom': cylinder, 'colour': 'grey', 
'opacity': 0.175, 'style': 'surface'},
                 }
            
bounds = {'line_width': 5, 'color': 'red', 'opacity': 0.8}

plot_simulation_timestep(particle_mesh_initial, geoms_to_plot, scalar_to_plot='velocity')
plot_simulation_timestep(particle_mesh_initial, geoms_to_plot, scalar_to_plot='diameter')
# plot_simulation_timestep(particle_mesh_initial, geoms_to_plot, scalar_to_plot='density')
# plot_simulation_timestep(particle_mesh_initial, geoms_to_plot, scalar_to_plot='mass')
# plot_simulation_timestep(particle_mesh_initial, geoms_to_plot, scalar_to_plot='volume')
# plot_simulation_timestep(particle_mesh_initial, geoms_to_plot, scalar_to_plot='avg_velocity')


# %% Animate with slider

animate_particles(particle_files_list, geoms_to_plot, scalar_to_plot='diameter')
animate_particles(particle_files_list, geoms_to_plot, scalar_to_plot='velocity', clim=[0, 0.005])

# %%
