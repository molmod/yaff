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


from __future__ import division
from __future__ import print_function

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from yaff.analysis.biased_sampling import SumHills
from molmod.units import kjmol
from molmod.constants import boltzmann

from mtd import T

def get_fes():
    npoints = 51
    # Construct a regular 2D grid, spanning from -pi to +pi in both dimensions
    grid0 = np.linspace(-np.pi,np.pi,npoints,endpoint=False)
    grid1 = np.linspace(-np.pi,np.pi,npoints,endpoint=False)
    grid = np.zeros((grid0.shape[0]*grid1.shape[0],2))
    grid[:,0] = np.repeat(grid0, grid1.shape[0])
    grid[:,1] = np.tile(grid1, grid0.shape[0])
    mtd = SumHills(grid)
    mtd.load_hdf5('traj.h5')
    fes = mtd.compute_fes()
    # Reshape to rectangular grids
    grid = grid.reshape((grid0.shape[0],grid1.shape[0],2))
    fes = fes.reshape((grid0.shape[0],grid1.shape[0]))
    return grid, fes

def make_plot(grid, fes):
    # Free energy as a function of DihedAngle(4,6,8,14), by integrating over
    # other collective variable
    beta = 1.0/boltzmann/T
    fes_phi = -1./beta*np.log(np.sum(np.exp(-beta*fes), axis=1))
    fes_phi -= np.amin(fes_phi)
    plt.clf()
    plt.plot(grid[:,0,0], fes_phi/kjmol)
    plt.xlabel("$\phi\,[\mathrm{rad}]$")
    plt.ylabel("$F\,[\mathrm{kJ}\,\mathrm{mol}^{-1}]$")
    plt.savefig('fes_phi.png')

if __name__=='__main__':
    grid, fes = get_fes()
    make_plot(grid, fes)
