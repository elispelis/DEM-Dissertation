import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from helpers import read_gns_file


def create_particle_mesh(timestep_data, particle_unique_mask=None):
    if particle_unique_mask is None:
        particle_unique_mask = np.arange(timestep_data.num_particles)

    if timestep_data.position.size > 0:
        particle_mesh = pv.PolyData(timestep_data.position[particle_unique_mask, :])
        particle_mesh["radius"] = timestep_data.radius[particle_unique_mask]
        particle_mesh["diameter"] = particle_mesh["radius"] * 2
        particle_mesh["mass"] = timestep_data.mass[particle_unique_mask]
        particle_mesh["volume"] = timestep_data.volume[particle_unique_mask]
        particle_mesh["density"] = timestep_data.density[particle_unique_mask]
        particle_mesh["group"] = timestep_data.group[particle_unique_mask]
        particle_mesh["velocity_x"] = timestep_data.velocity[particle_unique_mask, 0]
        particle_mesh["velocity_y"] = timestep_data.velocity[particle_unique_mask, 1]
        particle_mesh["velocity_z"] = timestep_data.velocity[particle_unique_mask, 2]
        particle_mesh["velocity"] = np.linalg.norm(timestep_data.velocity[particle_unique_mask, :], axis=1)
        particle_mesh["ang_velocity_x"] = timestep_data.ang_velocity[particle_unique_mask, 0]
        particle_mesh["ang_velocity_y"] = timestep_data.ang_velocity[particle_unique_mask, 1]
        particle_mesh["ang_velocity_z"] = timestep_data.ang_velocity[particle_unique_mask, 2]
        particle_mesh["ang_velocity"] = np.linalg.norm(timestep_data.ang_velocity[particle_unique_mask, :], axis=1)
        particle_mesh["avg_velocity_x"] = timestep_data.avg_velocity[particle_unique_mask, 0]
        particle_mesh["avg_velocity_y"] = timestep_data.avg_velocity[particle_unique_mask, 1]
        particle_mesh["avg_velocity_z"] = timestep_data.avg_velocity[particle_unique_mask, 2]
        particle_mesh["avg_velocity"] = np.linalg.norm(timestep_data.avg_velocity[particle_unique_mask, :], axis=1)
        particle_mesh["avg_acceleration_x"] = timestep_data.avg_acceleration[particle_unique_mask, 0]
        particle_mesh["avg_acceleration_y"] = timestep_data.avg_acceleration[particle_unique_mask, 1]
        particle_mesh["avg_acceleration_z"] = timestep_data.avg_acceleration[particle_unique_mask, 2]
        particle_mesh["avg_acceleration"] = np.linalg.norm(timestep_data.avg_acceleration[particle_unique_mask, :], axis=1)
        particle_mesh["torque_x"] = timestep_data.torque[particle_unique_mask, 0]
        particle_mesh["torque_y"] = timestep_data.torque[particle_unique_mask, 1]
        particle_mesh["torque_z"] = timestep_data.torque[particle_unique_mask, 2]
        particle_mesh["torque"] = np.linalg.norm(timestep_data.torque[particle_unique_mask, :], axis=1)
    else:
        particle_mesh = pv.PolyData(timestep_data.position)
        particle_mesh["radius"] = timestep_data.radius
        particle_mesh["diameter"] = particle_mesh["radius"] * 2

    return particle_mesh



def plot_simulation_timestep(particle_mesh, geoms_to_plot, bounds=None, scalar_to_plot=None, colourmap='viridis', 
clim=None, particle_mono_colour=None, theme_to_use='document', glyph_res=8):

    # Change plotting theme
    # pv.set_plot_theme("default")  ## grey background
    pv.set_plot_theme(theme_to_use)  ## white background
    
    # Low resolution geometry
    geom = pv.Sphere(theta_resolution=glyph_res, phi_resolution=glyph_res)

    # Progress bar is a new feature on master branch
    glyphed_particle_mesh = particle_mesh.glyph(scale="diameter", geom=geom, progress_bar=True)

    # create Plot
    print("\nRendering...")
    plotter = BackgroundPlotter()
    # plotter = pv.Plotter(notebook=False)

    # Add geometries
    for surface in geoms_to_plot:
        plotter.add_mesh(geoms_to_plot[surface]['geom'],
                        color=geoms_to_plot[surface]['colour'],
                        opacity=geoms_to_plot[surface]['opacity'],
                        style=geoms_to_plot[surface]['style'])

    # add particle data
    if len(glyphed_particle_mesh.array_names) > 0:
        if particle_mono_colour is not None:
            plotter.add_mesh(glyphed_particle_mesh, color=particle_mono_colour, smooth_shading=True, scalars=None, clim=clim)
        else:
            plotter.add_mesh(glyphed_particle_mesh, scalars=scalar_to_plot, smooth_shading=True, cmap=colourmap, clim=clim)

    if bounds is not None:
        plotter.add_bounding_box(line_width=5, color='black')

    plotter.show_axes()
    plotter.show()


class SceneEngine:
    def __init__(self, mesh, particle_files_list, glyph_res):
        self.output = mesh  # Expected PyVista mesh type
        self.particle_files_list = particle_files_list
        # default parameters
        self.kwargs = {
            'timestep': 1,
            'theta_resolution': glyph_res,
            'phi_resolution': glyph_res,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        # Low resolution geometry
        geom = pv.Sphere(theta_resolution=self.kwargs['theta_resolution'], phi_resolution=self.kwargs['phi_resolution'])

        # Progress bar is a new feature on master branch
        timestep_data = read_gns_file(self.particle_files_list[self.kwargs['timestep']])
        particle_mesh = create_particle_mesh(timestep_data)
        glyphed_particle_mesh = particle_mesh.glyph(scale="diameter", geom=geom, progress_bar=False)
        self.output.copy_from(glyphed_particle_mesh)
        return



def animate_particles(particle_files_list, geoms_to_plot, bounds=None, scalar_to_plot=None, colourmap='viridis', 
clim=None, particle_mono_colour=None, theme_to_use='document', glyph_res=8, notebook=False):

    num_timesteps = len(particle_files_list)

    # Change plotting theme
    pv.set_plot_theme(theme_to_use)  ## white background

    # setup initial particle mesh
    geom = pv.Sphere(theta_resolution=glyph_res, phi_resolution=glyph_res)
    timestep_data = read_gns_file(gns_particle_file=particle_files_list[1])
    particle_mesh = create_particle_mesh(timestep_data)
    starting_mesh = particle_mesh.glyph(scale="diameter", geom=geom, progress_bar=False)

    # create update engine
    engine = SceneEngine(starting_mesh, particle_files_list, glyph_res)

    # create Plot
    print("\nRendering...")  
    if notebook:
        plotter = pv.Plotter(notebook=True)
    else:
        plotter = BackgroundPlotter()

    # Add geometries
    for surface in geoms_to_plot:
        plotter.add_mesh(geoms_to_plot[surface]['geom'],
                        color=geoms_to_plot[surface]['colour'],
                        opacity=geoms_to_plot[surface]['opacity'],
                        style=geoms_to_plot[surface]['style'])
                        
    # add particle mesh
    if len(starting_mesh.array_names) > 0:
        plotter.add_mesh(starting_mesh, show_edges=False, scalars=scalar_to_plot, 
        smooth_shading=True, cmap=colourmap, clim=clim)

    plotter.add_slider_widget(
        callback=lambda value: engine('timestep', int(value)),
        rng=[0, num_timesteps-1],
        value=0,
        title="Timestep",
        fmt="%5.0f",
        pointa=(0.67, 0.9),
        pointb=(0.98, 0.9),
        style='modern',
    )
    plotter.show()