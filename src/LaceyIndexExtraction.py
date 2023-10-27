from edempy import Deck
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import os.path
import csv
import os
from LaceyClass import LaceyMixingAnalyzer  # Import the class from your module

sim_name = "Rot_drum"
sim_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', "rot_drum", "JKR_periodic_clean", str(sim_name)+".dem"))

with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Lacey_settings.txt")), 'r') as file:
    preferences = file.readlines()
    minCoords = np.array([float(i) for i in str(preferences[1]).split(',')])
    maxCoords = np.array([float(i) for i in str(preferences[3]).split(',')])
    bins = np.array([int(i) for i in str(preferences[5]).split(',')])
    cut_off = float(preferences[7])
    plot = str(preferences[9])
    file.close()
    settings = True

sim_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', "rot_drum", "JKR_periodic_clean", "Rot_drum.dem"))
lacey = LaceyMixingAnalyzer(minCoords, maxCoords, bins, sim_path)

Lacey_index = np.zeros(lacey.deck.numTimesteps)
b_coords, div_size = lacey.grid()

print("-------------------------------------------------------")
print("Processing: " + str(sim_name) + ".dem")
print("-------------------------------------------------------")

for i in range(10, lacey.deck.numTimesteps):
        particles = lacey.get_particles(i, np.array(lacey.deck.timestep[0].h5ParticleTypes))

        mass_1, mass_2, conc = lacey.bining(b_coords, div_size, particles, cut_off)

        Lacey_index[i] = lacey.Lacey(mass_1, mass_2, conc, cut_off, len(particles))
        print("Timestep: " + str(lacey.deck.timestepValues[i]) + " (s)")


        if i == lacey.deck.numTimesteps - 1: 
            if plot == 'yes\n':
                time = lacey.deck.timestepValues
                lacey.render(particles, b_coords, sim_name, Lacey_index, time, div_size, sim_path)
            lacey.write_out(Lacey_index, time, sim_path, sim_name)

print("-------------------------------------------------------")
print("Processing complete")
print("-------------------------------------------------------")
