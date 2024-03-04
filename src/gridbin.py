#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module documentation goes here

Created at 12:31, 03 Mar, 2024
"""

__author__ = 'J.P. Morrissey'
__copyright__ = 'Copyright 2024, J.P. Morrissey'
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = 'J.P. Morrissey'
__email__ = 'j.morrissey@ed.ac.uk'
__status__ = 'Development'

import functools

# Standard Library


# Imports
import numpy as np
import bottleneck as bn
from numba import njit

from src.helpers import generate_random_velocity, logical_or_func_reduce


# Local Sources




def find_bin_outliers(binned_indices, data, nbin):

    bin_ids = np.zeros(data.shape[0], dtype=np.int32)

    # find outliers
    outlier_mask_1 = np.sum(binned_indices == 0, axis=1)
    outlier_mask_2 = binned_indices[:, 0] == nbin[0] - 1
    outlier_mask_3 = binned_indices[:, 1] == nbin[1] - 1
    outlier_mask_4 = binned_indices[:, 2] == nbin[2] - 1
    # logical or
    # mask = (outlier_mask_1 + outlier_mask_2 + outlier_mask_3 + outlier_mask_4).astype(np.bool_)
    mask = logical_or_func_reduce(outlier_mask_1, outlier_mask_2, outlier_mask_3, outlier_mask_4)

    bin_ids[mask] = -1

    return bin_ids, mask



@njit(cache=True)
def _get_bin_index(data, edges):
    N, n_dims = data.shape

    # calculate the bin index for each element
    n_count = []
    for i in range(n_dims):
        n_count.append(np.searchsorted(edges[i], data[:, i], side='right'))

    for i in range(n_dims):
        # Find which points are on the rightmost edge.
        on_edge = (data[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        n_count[i][on_edge] -= 1

    binned_indices = np.stack((n_count[0], n_count[1], n_count[2])).T

    return binned_indices


class GridBin:
    """
    GridBin class for use with EDEMpy.

    Discretises domain into the specified number of bins in each direction. Every bin is of a equal dimension in a
    given direction and dimensions are automatically calculated based on the number of bins required.

    Attributes
    ----------
    dMin : array / list
        list of x,y,z coords to be used as minimum bin values in each direction.
    dMax : array / list
        list of x,y,z coords to be used as maximum bin value in each direction.
    xBins : int, default = 1
        number of bins in x direction
    yBins : int, default = 1
        number of bins in y direction
    zBins : int, default = 1
        number of bins in z direction

    Returns
    -------
    GridBin Object.

    """

    def __init__(self, dmin, dmax, xBins=1, yBins=1, zBins=1):
        """ Constructor for GridBin class.

        Parameters
        ----------
        dMin : array / list
            list of x,y,z coords to be used as minimum bin values in each direction.
        dMax : array / list
            list of x,y,z coords to be used as maximum bin values in each direction.
        xBins : int, default = 1
            number of bins in x direction
        yBins : int, default = 1
            number of bins in y direction
        zBins : int, default = 1
            number of bins in z direction

        """

        self.dMin = np.array(dmin)
        self.dMax = np.array(dmax)

        # parse the overloaded bins argument
        if np.ndim(xBins) == 0 and np.ndim(yBins) == 0 and np.ndim(zBins) == 0:
            # all equal sized bins
            self.xBins = xBins
            self.yBins = yBins
            self.zBins = zBins

            # get edges
            self._getBinEdges(xBins=xBins, yBins=yBins, zBins=zBins)

        elif np.ndim(xBins) == 1 and np.ndim(yBins) == 1 and np.ndim(zBins) == 1:
            # list of edges given in each direction
            self.xBins = len(xBins) - 1
            self.yBins = len(yBins) - 1
            self.zBins = len(zBins) - 1

            # set edges
            self.xBinEdges = xBins
            self.yBinEdges = yBins
            self.zBinEdges = zBins

        else:
            # edges provided in one direction, number in other
            self._getBinEdges(xBins=xBins, yBins=yBins, zBins=zBins)
            self.xBins = len(self.xBinEdges) - 1
            self.yBins = len(self.yBinEdges) - 1
            self.zBins = len(self.zBinEdges) - 1

        self.numBins = self.xBins * self.yBins * self.zBins
        self._numBinEdges = np.array([self.xBins, self.yBins, self.zBins]) + 1
        self.grid_shape = (self.xBins, self.yBins, self.zBins)

        # get centres
        self.xBinCentres = self._getBinCentres(self.xBinEdges)
        self.yBinCentres = self._getBinCentres(self.yBinEdges)
        self.zBinCentres = self._getBinCentres(self.zBinEdges)

        # get dims
        self.xBinDims = np.diff(self.xBinEdges)
        self.yBinDims = np.diff(self.yBinEdges)
        self.zBinDims = np.diff(self.zBinEdges)

        self._get_bin_sizes()

        # internal bin and edge count for binninn algorithm
        self.__nbins = np.array([self.xBins, self.yBins, self.zBins], np.intp) + 2
        self.__edges = (self.xBinEdges, self.yBinEdges, self.zBinEdges)

    def _getBinEdges(self, xBins=1, yBins=1, zBins=1):
        """
        Returns np.ndarray of bin edge values for chosen axis

        Parameters
        ----------
        xBins : int
            amount of bin divisions to split the data into in the x-direction.
        yBins : int
            amount of bin divisions to split the data into in the y-direction.
        zBins : int
            amount of bin divisions to split the data into in the z-direction.

        Returns
        -------
        xBinEdges : np.ndarray
            Array of bin edges in x-direction. Includes both outer edges.
        yBinEdges : np.ndarray
            Array of bin edges in y-direction. Includes both outer edges.
        zBinEdges : np.ndarray
            Array of bin edges in z-direction. Includes both outer edges.

        """

        if np.ndim(xBins) == 0:
            self.xBinEdges = np.linspace(self.dMin[0], self.dMax[0], xBins + 1, endpoint=True)
        else:
            self.xBinEdges = xBins

        if np.ndim(yBins) == 0:
            self.yBinEdges = np.linspace(self.dMin[1], self.dMax[1], yBins + 1, endpoint=True)
        else:
            self.yBinEdges = yBins

        if np.ndim(zBins) == 0:
            self.zBinEdges = np.linspace(self.dMin[2], self.dMax[2], zBins + 1, endpoint=True)
        else:
            self.zBinEdges = zBins

    def _get_bin_sizes(self):
        """
        Returns bin sizes in each dimension.

        All bins in the grid are of equal dimensions.

        Parameters
        ----------
        xBins : int
            amount of bin divisions to split the data into in the x-direction.
        yBins : int
            amount of bin divisions to split the data into in the y-direction.
        zBins : int
            amount of bin divisions to split the data into in the z-direction.

        Returns
        -------
        list
            Returns the dimensions of each bin in the x, y and z directions.

        """

        # calculated interval step for bin edges
        self.avgBinDimensions = (self.dMax - self.dMin) / np.array([self.xBins, self.yBins, self.zBins])

    @staticmethod
    def _getBinCentres(axisBinEdges):
        """Returns numpy.ndarray of 1D coordinates for the center of a given list of bin edges.

        Parameters
        ----------
        axisBinEdges : np.ndarray
            np.ndarray of bin edge values (low value starting edge of each bin)

        Returns
        -------
        np.ndarray
             Array of the centre of a bin for a given set of bin edges.

        """
        # Add half the difference between edges to the left edges of the bin to get the centre
        return axisBinEdges[:-1] + np.diff(axisBinEdges) / 2

    def get_bin_indices(self, data, outliers_present=True):
        binned_indices = _get_bin_index(data, self.__edges)

        if outliers_present:
            bin_ids, outlier_mask = find_bin_outliers(binned_indices, data, self._GridBin__nbins)
            binned_indices -= 1
            bin_ids[~outlier_mask] = np.ravel_multi_index(binned_indices[~outlier_mask].T,
                                                          tuple(data, self.__nbins - 2))

        else:
            outlier_mask = None
            binned_indices -= 1
            bin_ids = np.ravel_multi_index(binned_indices.T, tuple(self.__nbins - 2))

        return binned_indices, bin_ids, outlier_mask

    def get_bin_index(self, bin_id):
        return np.unravel_index(bin_id, self.grid_shape)


    def get_binned_data(self, coordinates, binning_data):

        bin_index, bin_id, outlier_mask = self.get_bin_indices(coordinates, outliers_present=False)

        unique_bins, unique_indices = np.unique(bin_id, return_index=True)
        non_outliers = np.where(unique_bins >= 0)[0]
        unique_bins = unique_bins[non_outliers]
        unique_indices = bin_index[unique_indices[non_outliers]]

        binned_data = [[[None for k in range(self.zBins)] for j in range(self.yBins)] for i in range(self.xBins)]

        # do binning
        for bin, index in zip(unique_bins, unique_indices):
            x, y, z = index
            mask = (bin_id == bin)
            if binning_data.ndim == 1:
                binned_data[x][y][z] = binning_data[mask]
            elif binning_data.ndim == 2:
                binned_data[x][y][z] = binning_data[mask, :]

        return binned_data



    def calculate_velocity_stdev(self, coordinates, velocity):

        bin_index, bin_id, outlier_mask = self.get_bin_indices(coordinates, outliers_present=False)

        unique_bins, unique_indices = np.unique(bin_id, return_index=True)
        non_outliers = np.where(unique_bins >= 0)[0]
        unique_bins = unique_bins[non_outliers]
        unique_indices = bin_index[unique_indices[non_outliers]]

        binned_data = [[[None for k in range(self.zBins)] for j in range(self.yBins)] for i in range(self.xBins)]
        velocities = np.zeros((self.numBins, 3))
        mean_fluctuating_velocities = np.zeros((self.numBins, 3))

        # do binning
        for bin, index in zip(unique_bins, unique_indices):
            x, y, z = index
            mask = (bin_id == bin)
            velocities[bin, :] = np.mean(velocity[mask, :], axis=0)
            binned_data[x][y][z] = np.flatnonzero(mask)

            # fluctuating velocity calculation
            deviations = velocity[mask, :] - velocities[bin, :]
            mean_squared_deviation = np.mean(deviations**2, axis=0)
            mean_fluctuating_velocities[bin, :] = np.sqrt(mean_squared_deviation)

        return velocities, mean_fluctuating_velocities, binned_data


    def apply_random_velocity(self, coordinates, velocity_stds, t_rnn):

        bin_index, bin_id, outlier_mask = self.get_bin_indices(coordinates, outliers_present=False)

        unique_bins, unique_indices = np.unique(bin_id, return_index=True)
        non_outliers = np.where(unique_bins >= 0)[0]
        unique_bins = unique_bins[non_outliers]
        unique_indices = bin_index[unique_indices[non_outliers]]

        binned_data = [[[None for k in range(self.zBins)] for j in range(self.yBins)] for i in range(self.xBins)]
        perturbed_positions = np.zeros_like(coordinates)

        # do binning
        for bin, index in zip(unique_bins, unique_indices):
            x, y, z = index
            mask = (bin_id == bin)
            perturbed_positions[mask, :] = coordinates[mask, :] + generate_random_velocity(velocity_stds[bin], mask.sum()) * t_rnn
            binned_data[x][y][z] = np.flatnonzero(mask)

        return perturbed_positions, binned_data

    def get_particle_concentration(self, coordinates, binning_data):

        bin_index, bin_id, outlier_mask = self.get_bin_indices(coordinates, outliers_present=False)

        unique_bins, unique_indices = np.unique(bin_id, return_index=True)
        non_outliers = np.where(unique_bins >= 0)[0]
        unique_bins = unique_bins[non_outliers]
        unique_indices = bin_index[unique_indices[non_outliers]]

        binned_data = [[[None for k in range(self.zBins)] for j in range(self.yBins)] for i in range(self.xBins)]

        # do binning
        for bin, index in zip(unique_bins, unique_indices):
            x, y, z = index
            mask = (bin_id == bin)
            mass_1 = bn.nansum(binning_data[mask] == 0)
            mass_2 = bn.nansum(binning_data[mask] == 1)
            concentration =  mass_1 / (mass_1 + mass_2)

            binned_data[x][y][z] = (mass_1, mass_2, concentration)

        return binned_data