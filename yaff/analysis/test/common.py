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


import h5py as h5
from contextlib import contextmanager
import numpy as np

from molmod.test.common import tmpdir
from yaff import *
from yaff.sampling.test.common import get_ff_water32
from yaff.test.common import get_alaninedipeptide_amber99ff


@contextmanager
def run_nve_water32(suffix, prefix):
    # Work in a temporary directory
    with tmpdir(suffix, prefix) as dn_tmp:
        # Setup a test FF
        ff = get_ff_water32()
        # Run a test simulation
        with h5.File('%s/output.h5' % dn_tmp) as f:
            hdf5 = HDF5Writer(f)
            nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=hdf5)
            nve.run(5)
            assert nve.counter == 5
            yield dn_tmp, nve, f


@contextmanager
def run_nvt_water32(suffix, prefix):
    # Work in a temporary directory
    with tmpdir(suffix, prefix) as dn_tmp:
        # Setup a test FF
        ff = get_ff_water32()
        # Run a test simulation
        with h5.File('%s/output.h5' % dn_tmp) as f:
            hdf5 = HDF5Writer(f)
            thermostat = LangevinThermostat(temp=300)
            nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[hdf5, thermostat])
            nvt.run(5)
            assert nvt.counter == 5
            yield dn_tmp, nvt, f


@contextmanager
def run_opt_water32(suffix, prefix):
    # Work in a temporary directory
    with tmpdir(suffix, prefix) as dn_tmp:
        # Setup a test FF
        ff = get_ff_water32()
        # Run a test simulation
        with h5.File('%s/output.h5' % dn_tmp) as f:
            hdf5 = HDF5Writer(f)
            opt = CGOptimizer(FullCellDOF(ff), hooks=hdf5)
            opt.run(5)
            assert opt.counter == 5
            yield dn_tmp, opt, f


@contextmanager
def run_mtd_alanine(suffix, prefix):
    # Work in a temporary directory
    with tmpdir(suffix, prefix) as dn_tmp:
        # MTD settings
        sigmas = np.array([0.35*rad,0.35*rad])
        pace = 4
        K = 1.2*kjmol
        # Construct metadynamics as a Yaff hook
        ff = get_alaninedipeptide_amber99ff()
        cv0 = CVInternalCoordinate(ff.system, DihedAngle(4,6,8,14))
        cv1 = CVInternalCoordinate(ff.system, DihedAngle(6,8,14,16))
        # Dihedral angles are periodic, this has to be taken into account!
        periodicities = np.array([2.0*np.pi,2.0*np.pi])
        # Run a test simulation
        with h5.File('%s/output.h5' % dn_tmp) as f:
            hdf5 = HDF5Writer(f)
            mtd = MTDHook(ff, [cv0,cv1], sigmas, K, f=f, start=pace, step=pace,
                periodicities=periodicities)
            nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=[mtd])
            nve.run(12)
            yield dn_tmp, nve, f
