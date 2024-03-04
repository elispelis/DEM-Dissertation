#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module documentation goes here

Created at 12:01, 04 Mar, 2024
"""


__author__ = 'J.P. Morrissey'
__copyright__ = 'Copyright 2024, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = 'J.P. Morrissey'
__email__ = 'j.morrissey@ed.ac.uk'
__status__ = '{dev_status}'

import bottleneck as bn
import numpy as np


# Standard Library


# Imports


# Local Sources

def fix_particle_coords(local_mean, drum_r, drum_w):

    violating_particles = np.where(abs(local_mean[:,1])>drum_w)[0]
    side_fixed = len(violating_particles)
    local_mean[violating_particles, 1] = np.sign(local_mean[violating_particles, 1]) * drum_w

    distances = np.sqrt((local_mean[:, 0])**2+(local_mean[:, 2])**2)
    radial_violating_particles = np.where(distances>drum_r)[0]
    profile_fixed = len(radial_violating_particles)

    norm_factor = drum_r / distances[radial_violating_particles]
    local_mean[radial_violating_particles, 0] *= norm_factor
    local_mean[radial_violating_particles, 2] *= norm_factor

    print(f"Fixed {profile_fixed+side_fixed} particle(s). (Profile: {profile_fixed}, Side: {side_fixed})" )

    return local_mean


def unpack_mixing_results(grid_bin, binned_masses):

    mass_1 = []
    mass_2 = []
    concentration = []

    for bin in range(grid_bin.numBins):
        x, y, z = grid_bin.get_bin_index(bin)
        res = binned_masses[x][y][z]
        if res is not None:
            mass_1.append(res[0])
            mass_2.append(res[1])
            concentration.append(res[2])

    return np.array(mass_1), np.array(mass_2), np.array(concentration)


def Lacey(mass_1, mass_2, conc, cut_off, p_num):
    P = bn.nansum(mass_1) / (bn.nansum(mass_1) + bn.nansum(mass_2))
    S0 = P * (1 - P)
    mass = mass_1 + mass_2
    index = np.where(mass > cut_off)
    nBins = len(mass[index])
    S = bn.nansum((conc[index] - P)**2) / nBins
    Sr = S0 / (p_num / nBins)
    M = (S0 - S) / (S0 - Sr)

    return M


def generate_random_velocity(std_deviation, num_samples=1):
    x_vel_random = np.random.normal(loc=0, scale=std_deviation[0], size=num_samples)
    y_vel_random = np.random.normal(loc=0, scale=std_deviation[1], size=num_samples)
    z_vel_random = np.random.normal(loc=0, scale=std_deviation[2], size=num_samples)

    return np.array((x_vel_random, y_vel_random, z_vel_random)).T


def logical_or_func_reduce(*l):
    """
    Recursive function to do logical_or on multiple conditions. Any number of conditions can be provided.


    Parameters
    ----------
    l : array-like
        number of array_like conditions on which to perform logical_or operation

    Returns
    -------
    out : array-like
        boolean array

    """
    return functools.reduce(np.logical_or, l)