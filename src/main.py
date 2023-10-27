import matplotlib.pyplot as plt
import os
from LaceyClass import LaceyMixingAnalyzer
from extrapolation import extrapolation

if __name__ == "__main__":
    simulation = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', "rot_drum", "JKR_periodic_clean", "Rot_drum.dem"))
    sim_path = os.path.dirname(simulation)

    start_t = 1
    end_t = 30

    extrap = extrapolation(start_t, end_t, simulation)
    id_dict = extrap.id_dictionary(extrap.sim_time[0])
    init_particles = extrap.get_particle_coords(extrap.sim_time[0])
    plt.figure()
    extrap.plot_particles(init_particles, id_dict)
    pos_dict = extrap.import_dict(sim_path, "peak23")
    plt.figure()
    extrap.plot_particles(extrap.extrapolate_particles(extrap.get_particle_coords(43),pos_dict),id_dict)
    plt.show()
    print("Done!")