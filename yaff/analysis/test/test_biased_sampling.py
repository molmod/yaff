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

import shutil, os
import pkg_resources
import h5py
import numpy as np

from yaff import *
from yaff.analysis.test.common import run_mtd_alanine
from molmod.units import kjmol


def test_sum_hills_alanin():
    npoints = 10
    # Construct a regular 2D grid
    grid0 = np.linspace(-3.0,2.0,npoints)
    grid1 = np.linspace(2.0,3.0,npoints)
    grid = np.zeros((grid0.shape[0]*grid1.shape[0],2))
    grid[:,0] = np.repeat(grid0, grid1.shape[0])
    grid[:,1] = np.tile(grid1, grid0.shape[0])
    # Run short MTD simulation
    with run_mtd_alanine(__name__, 'test_mtd') as (dn_tmp, nvt, fh5):
        # Postprocessing
        mtd = SumHills(grid)
        mtd.load_hdf5(os.path.join(dn_tmp,'output.h5'))
        fes = mtd.compute_fes()
        # Manual check
        q0s = fh5['hills/q0'][:]
        Ks = fh5['hills/K'][:]
        sigmas = fh5['hills/sigma'][:]
        periodicities = fh5['hills/periodicities'][:]
        for igrid in range(grid.shape[0]):
            f = 0.0
            for ihill in range(q0s.shape[0]):
                deltas = grid[igrid]-q0s[ihill]
                deltas -= np.floor(0.5+deltas/periodicities)*periodicities
                f -= Ks[ihill]*np.exp(-np.sum(deltas**2/2.0/sigmas**2))
            assert np.abs(f-fes[igrid])<1e-10*kjmol
