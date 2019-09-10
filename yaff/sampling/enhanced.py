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
'''Support for enhanced sampling methods'''


from __future__ import division

import numpy as np
import h5py as h5

from molmod.constants import boltzmann

from yaff.pes.bias import GaussianHills
from yaff.pes.ff import ForcePartBias
from yaff.sampling.iterative import Hook
from yaff.log import log, timer


__all__ = [
    'MTDHook',
]


class MTDHook(Hook):
    """
    Metadynamics simulations
    """
    def __init__(self, ff, cv, sigma, K, f=None, start=0, step=1,
                 restart_file=None, tempering=0, periodicities=None):
        """
           **Arguments:**

           ff
                A ForceField instance

           cv
                A single ``CollectiveVariable`` or a list of ``CollectiveVariable``
                instances.

           sigma
                The width of the Gaussian or a NumPy array [Ncv] specifying the
                widths of the Gaussians

           K
                The prefactor of the Gaussian hills.

           **Optional arguments:**

           f
                A h5.File object to write the Gaussian hills to.

           start
                The first iteration at which a Gaussian hill should be added.

           step
                A Gaussian hill will be added every `step` iterations.

           restart_files
                A single h5 file containing hills added during previous
                simulations. Gaussian hills present in restart_files will
                be added at the start of the simulation.

           tempering
                Perform a well-tempered metadynamics simulation

           periodicities
                The periodicity of the single collective variable or a [Ncv]
                NumPy array specifying the periodicity of each
                collective variable. Specifying None means the CV is not
                periodic.
        """
        self.hills = GaussianHills(cv, sigma, periodicities=periodicities)
        self.K = K
        self.f = f
        self.tempering = tempering
        # Add the bias part to the force field
        part = ForcePartBias(ff.system)
        part.add_term(self.hills)
        ff.add_part(part)
        # Initialize hills from restart files
        if restart_file is not None:
            if not 'hills' in restart_file:
                raise ValueError("Could not read hills group from %s"%(restart_file))
            if not self.tempering==restart_file['hills'].attrs['tempering']:
                raise ValueError("Inconsistent tempering between runs")
            if not np.all(self.hills.periodicities==restart_file['hills/periodicities']):
                raise ValueError("Inconsistent periodicities between runs")
            q0 = restart_file['hills/q0'][:]
            K = restart_file['hills/K'][:]
            self.hills.add_hills(q0, K)
            if self.f is not None:
                for istep in range(q0.shape[0]):
                    self.dump_h5(q0[istep], K[istep])
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        # Compute current CV values
        q0s = self.hills.calculate_cvs()
        # Compute force constant
        K = self.K
        if self.tempering!=0.0:
            K *= np.exp(-self.hills.compute()/boltzmann/self.tempering)
        # Add a hill
        self.hills.add_hill(q0s, K)
        if self.f is not None:
            self.dump_h5(q0s, K)

    def dump_h5(self, q0s, K):
        # Write to HDF5 file
        if 'hills' not in self.f:
            self.init_hills()
        hgrp = self.f['hills']
        # Determine the row to write the current iteration to.
        row = min(hgrp[key].shape[0] for key in ['q0','K'] if key in hgrp.keys())
        for label, data in zip(['q0','K'], [q0s, K]):
            ds = hgrp[label]
            if ds.shape[0] <= row:
                # Do not over-allocate. hdf5 works with chunks internally.
                ds.resize(row+1, axis=0)
            ds[row] = data

    def init_hills(self):
        hgrp = self.f.create_group('hills')
        hgrp.create_dataset('sigma', data=self.hills.sigmas)
        hgrp.create_dataset('q0', (0,self.hills.ncv), maxshape=(None,self.hills.ncv), dtype=float)
        hgrp.create_dataset('K', (0,), maxshape=(None,), dtype=float)
        hgrp.attrs['tempering'] = self.tempering
        if self.hills.periodicities is not None:
            hgrp.create_dataset('periodicities', data=self.hills.periodicities)
