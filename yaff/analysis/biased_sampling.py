# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
'''Process biased sampling methods'''


from __future__ import division

import numpy as np
import h5py as h5

from yaff.log import log


__all__ = ['SumHills']


class SumHills(object):
    def __init__(self, grid):
        """
           Computes a free energy profile by summing hills deposited during
           a metadyanmics simulation.

           **Argument:**

           grid
                A [N, n] NumPy array, where n is the number of collective
                variables and N is the number of grid points
        """
        self.grid = grid
        self.ncv = self.grid.shape[1]
        self.q0s = None

    def compute_fes(self):
        if self.q0s is None:
            raise ValueError("Hills not initialized")
        ngauss = self.q0s.shape[0]
        if self.tempering != 0.0:
            prefactor = self.tempering/(self.tempering+self.T)
        else: prefactor = 1.0
        # Compute exponential argument
        exparg = np.diagonal(np.subtract.outer(self.grid, self.q0s), axis1=1, axis2=3)**2
        exparg = np.multiply(exparg, 0.5/self.sigmas**2)
        exparg = np.sum(exparg, axis=2)
        exponents = np.exp(-exparg)
        # Compute the bias energy
        fes = -prefactor*np.sum(np.multiply(exponents, self.Ks),axis=1)
        return fes

    def set_hills(self, q0s, Ks, sigmas, tempering=0.0, T=None):
        # Safety checks
        assert q0s.shape[1]==self.ncv
        assert sigmas.shape[0]==self.ncv
        assert q0s.shape[0]==Ks.shape[0]
        if tempering != 0.0 and T is None:
            raise ValueError("For a well-tempered MTD run, the temperature "
                "has to be specified")
        self.q0s = q0s
        self.sigmas = sigmas
        self.Ks = Ks
        self.tempering = tempering
        self.T = T

    def load_hdf5(self, fn, T=None):
        """
           Read information from HDF5 file

           **Arguments:**

           fn
                A HDF5 filename containing a hills group. If this concerns a well-
                tempered MTD run, the simulation temperature should be provided
                Otherwise, it will be read from the HDF5 file.
        """
        with h5.File(fn,'r') as f:
            q0s = f['hills/q0'][:]
            Ks = f['hills/K'][:]
            sigmas = f['hills/sigma'][:]
            tempering = f['hills'].attrs['tempering']
            if tempering!=0.0:
                if T is None:
                    if not 'trajectory/temp' in f:
                        raise ValueError("For a well-tempered MTD run, the temperature "
                            "should be specified or readable from the trajectory/temp "
                            "group in the HDF5 file")
                    T = np.mean(f['trajectory/temp'][:])
                if log.do_medium:
                    log("Well-tempered MTD run: T = %s deltaT = %s"%(log.temperature(T), log.temperature(tempering)))
            self.set_hills(q0s, Ks, sigmas, tempering=tempering, T=T)
