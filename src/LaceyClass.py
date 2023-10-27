# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:41:33 2022

@author: spantaleev, elisbright
"""

from edempy import Deck
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import os.path
import csv

class LaceyMixingAnalyzer:
    def __init__(self, minCoords, maxCoords, Bins, deck):
        self.minCoords = minCoords
        self.maxCoords = maxCoords
        self.Bins = Bins
        self.deck = Deck(deck)

    def grid(self):
        lengths = self.maxCoords - self.minCoords
        resolution = np.array(self.Bins)
        divisionSize = lengths / resolution
        xCoords = np.arange(self.minCoords[0] + (divisionSize[0] / 2), self.maxCoords[0], divisionSize[0])
        yCoords = np.arange(self.minCoords[1] + (divisionSize[1] / 2), self.maxCoords[1], divisionSize[1])
        zCoords = np.arange(self.minCoords[2] + (divisionSize[2] / 2), self.maxCoords[2], divisionSize[2])
        coords = []

        for i in range(0, len(xCoords)):
            for j in range(0, len(yCoords)):
                for k in range(0, len(zCoords)):
                    coords.append([xCoords[i], yCoords[j], zCoords[k]])

        return np.array(coords), divisionSize

    def get_particles(self, t_step, p_types):
        p_coords = np.zeros((1, 3))
        p_masses = np.zeros(1)
        p_types_ind = np.zeros(1)

        for n in p_types:
            p_coords = np.append(p_coords, self.deck.timestep[t_step].particle[int(n)].getPositions(), axis=0)
            p_masses = np.append(p_masses, self.deck.timestep[t_step].particle[int(n)].getMass(), axis=0)
            p_types_ind = np.append(p_types_ind, np.ones(len(self.deck.timestep[t_step].particle[int(n)].getMass())) * int(n), axis=0)

        p_coords = np.delete(p_coords, (0), axis=0)
        p_masses = np.delete(p_masses, (0), axis=0)
        p_types_ind = np.delete(p_types_ind, (0), axis=0)
        particles = np.column_stack((p_coords[:, 0], p_coords[:, 1], p_coords[:, 2], p_masses, p_types_ind))

        return particles

    def bining(self, b_coords, div_size, particles, cut_off):
        conc = np.zeros(len(b_coords))
        mass_1 = np.zeros(len(b_coords))
        mass_2 = np.zeros(len(b_coords))

        for i in range(len(b_coords)):
            mins = b_coords[i] - div_size / 2
            maxs = b_coords[i] + div_size / 2
            p_type_1 = particles[0, -1]
            index_1 = np.where((particles[:, 0] < maxs[0]) & (particles[:, 1] < maxs[1]) & (particles[:, 2] < maxs[2]) &
                               (particles[:, 0] > mins[0]) & (particles[:, 1] > mins[1]) & (particles[:, 2] > mins[2]) &
                               (particles[:, -1] == p_type_1))
            index_2 = np.where((particles[:, 0] < maxs[0]) & (particles[:, 1] < maxs[1]) & (particles[:, 2] < maxs[2]) &
                               (particles[:, 0] > mins[0]) & (particles[:, 1] > mins[1]) & (particles[:, 2] > mins[2]) &
                               (particles[:, -1] != p_type_1))
            mass_1[i] = np.sum(particles[index_1, 3])
            mass_2[i] = np.sum(particles[index_2, 3])
            if (mass_1[i] + mass_2[i]) > 0:
                conc[i] = mass_1[i] / (mass_1[i] + mass_2[i])

        return mass_1, mass_2, conc

    def Lacey(self, mass_1, mass_2, conc, cut_off, p_num):
        P = np.sum(mass_1) / (np.sum(mass_1) + np.sum(mass_2))
        S0 = P * (1 - P)
        mass = mass_1 + mass_2
        index = np.where(mass > cut_off)
        nBins = len(mass[index])
        S = np.sum(np.power((conc[index] - P), 2)) / nBins
        Sr = S0 / (p_num / nBins)
        M = (S0 - S) / (S0 - Sr)
        return M

    def render(self, particles, b_coords, name, Lacey_index, time, div_size, sim_path):
        fig_1 = plt.figure()
        ax = fig_1.add_subplot(111, projection='3d')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)
        index_1 = np.where(particles[:, -1] == particles[0, 4])
        index_2 = np.where(particles[:, -1] != particles[0, 4])
        
        ax.scatter(particles[index_1, 0], particles[index_1, 1], particles[index_1, 2], s=0.01, c="b")
        ax.scatter(particles[index_2, 0], particles[index_2, 1], particles[index_2, 2], s=0.01, c="r")
        
        x = np.unique(b_coords[:, 0])
        y = np.unique(b_coords[:, 1])
        z = np.unique(b_coords[:, 2])
        
        x = np.append(x - div_size[0] / 2, np.amax(x) + div_size[0] / 2)
        y = np.append(y - div_size[1] / 2, np.amax(y) + div_size[1] / 2)
        z = np.append(z - div_size[2] / 2, np.amax(z) + div_size[1] / 2)
        
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    mult_y = np.ones_like(y)
                    mult_z = np.ones_like(z)
                    mult_x = np.ones_like(x)
                    
                    ax.plot(x[i] * mult_y, y, mult_y * z[k], c='black', linewidth=0.03)
                    ax.plot(x, y[j] * mult_x, mult_x * z[k], c='black', linewidth=0.03)
                    ax.plot(x[i] * mult_z, y[j] * mult_z, z, c='black', linewidth=0.03)
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Z coordinate')
        ax.set_xlim([np.amin(b_coords), np.amax(b_coords)])
        ax.set_ylim([np.amin(b_coords), np.amax(b_coords)])
        ax.set_zlim([np.amin(b_coords), np.amax(b_coords)])
        plt.close(fig_1)
        fig1_name = str(name)+"_Bins.png"
        fig_1.savefig(os.path.join(os.path.dirname(sim_path), fig1_name), dpi=300)
        
        fig_2 = plt.figure()
        plt.plot(time, Lacey_index, 'b-')
        plt.xlabel('Time (s)', fontsize=14, color='black')
        plt.ylabel('Lacey mixing index', fontsize=14, color='black')
        plt.title("Lacey mixing index evolution for " + str(name))
        plt.grid(True)
        plt.close(fig_2)
        fig2_name = str(name)+"_Lacey_vs_Time.png"
        fig2_path = os.path.join(os.path.dirname(sim_path), fig2_name)
        fig_2.savefig(os.path.join(os.path.dirname(sim_path), fig2_name), dpi=300)

        print("Saved "+str(fig1_name)+" and "+str(fig2_name)+" into "+str(os.path.dirname(sim_path)))

    def write_out(self, Lacey_index, time, sim_path, sim_name):
        output = np.column_stack((time, Lacey_index))

        sim_path = os.path.dirname(sim_path)
        np.savetxt(os.path.join(sim_path, str(sim_name)+"_Report.csv"), output, delimiter=",")