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

import os

import pkg_resources
import h5py as h5
import numpy as np
import os

from yaff import *
from yaff.test.common import get_alaninedipeptide_amber99ff
from molmod.test.common import tmpdir


def test_mtd_alanine():
    # MTD settings
    sigma = 0.35*rad
    pace = 4
    K = 1.2*kjmol
    # Construct metadynamics as a Yaff hook
    ff = get_alaninedipeptide_amber99ff()
    cv = CVInternalCoordinate(ff.system, DihedAngle(4,6,8,14))
    with h5.File('yaff.sampling.test.test_enhanced.test_mtd_alanine.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f)
        mtd = MTDHook(ff, cv, sigma, K, f = f, start=pace, step=pace, periodicities=2*np.pi)
        nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[mtd])
        vel0 = nvt.vel.copy()
        nvt.run(12)
        # Check HDF5 output
        assert 'hills' in f
        assert np.all(f['hills/q0'][:]==mtd.hills.q0s)
        assert np.all(f['hills/K'][:]==mtd.hills.Ks)
        assert f['hills/sigma'].shape[0]==1
        assert f['hills/sigma'][0]==sigma
    # Same simulation using PLUMED
    with tmpdir(__name__, 'test_mtd_alanine') as dirname:
        # PLUMED input commands
        commands = "phi: TORSION ATOMS=5,7,9,15\n"
        commands += "metad: METAD ARG=phi PACE=%d HEIGHT=%f SIGMA=%f FILE=%s\n"\
            % (pace, K/kjmol, sigma/rad, os.path.join(dirname,'hills'))
        commands += "FLUSH STRIDE=1"
        # Write PLUMED commands to file
        fn = os.path.join(dirname, 'plumed.dat')
        with open(fn,'w') as f:
            f.write(commands)
        # Setup Plumed
        ff = get_alaninedipeptide_amber99ff()
        plumed = ForcePartPlumed(ff.system, fn=fn)
        ff.add_part(plumed)
        nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[plumed], vel0=vel0)
        nvt.run(12)
        hills = np.loadtxt(os.path.join(dirname,'hills'))
    # Compare hill centers
    assert np.all(np.abs(hills[:,1]-mtd.hills.q0s[:,0])<1e-10*rad)


def test_mtd_alanine_tempered():
    # MTD settings
    sigma = 0.35*rad
    pace = 4
    K = 1.2*kjmol
    tempering = 1800*kelvin
    # Construct metadynamics as a Yaff hook
    ff = get_alaninedipeptide_amber99ff()
    cv = CVInternalCoordinate(ff.system, DihedAngle(4,6,8,14))
    mtd = MTDHook(ff, cv, sigma, K, start=pace, step=pace, tempering=tempering, periodicities=2*np.pi)
    nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[mtd])
    vel0 = nvt.vel.copy()
    nvt.run(12)
    # Same simulation using PLUMED
    with tmpdir(__name__, 'test_mtd_alanine') as dirname:
        # PLUMED input commands
        commands = "phi: TORSION ATOMS=5,7,9,15\n"
        # Bias factor, some confusion about its definition in the PLUMED manual...
        gamma = (300.+tempering)/300.0
        commands += "metad: METAD ARG=phi PACE=%d HEIGHT=%f SIGMA=%f FILE=%s BIASFACTOR=%f TEMP=300.0\n"\
            % (pace, K/kjmol, sigma/rad, os.path.join(dirname,'hills'), gamma)
        commands += "FLUSH STRIDE=1"
        # Write PLUMED commands to file
        fn = os.path.join(dirname, 'plumed.dat')
        with open(fn,'w') as f:
            f.write(commands)
        # Setup Plumed
        ff = get_alaninedipeptide_amber99ff()
        plumed = ForcePartPlumed(ff.system, fn=fn)
        ff.add_part(plumed)
        nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[plumed], vel0=vel0)
        nvt.run(12)
        hills = np.loadtxt(os.path.join(dirname,'hills'))
    # Compare Gaussian heights, PLUMED writes reweighted heights
    Kref = hills[:,3]*tempering/(300+tempering)*kjmol
    assert np.all(np.abs(Kref-mtd.hills.Ks)<1e-5*kjmol)
    assert np.all(np.abs(hills[:,1]-mtd.hills.q0s[:,0])<1e-10*rad)


def test_mtd_restart():
    # MTD settings
    sigma = 0.35*rad
    pace = 4
    K = 1.2*kjmol
    # Construct metadynamics as a Yaff hook
    ff = get_alaninedipeptide_amber99ff()
    cv = CVInternalCoordinate(ff.system, DihedAngle(4,6,8,14))
    with h5.File('yaff.sampling.test.test_enhanced.test_mtd_restart.h5',
            driver='core', backing_store=False) as f0:
        hdf5 = HDF5Writer(f0)
        mtd = MTDHook(ff, cv, sigma, K, f = f0, start=pace, step=pace, periodicities=2*np.pi)
        nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[mtd])
        vel0 = nvt.vel.copy()
        nvt.run(12)
        ff = get_alaninedipeptide_amber99ff()
        with h5.File('yaff.sampling.test.test_enhanced.test_mtd_restarted.h5',
                driver='core', backing_store=False) as f1:
            mtd_restart = MTDHook(ff, cv, sigma, K, f=f1, start=pace,
                step=pace, restart_file=f0, periodicities=2*np.pi)
            nvt = VerletIntegrator(ff, 1.0*femtosecond, hooks=[mtd_restart])
            nvt.run(12)
            assert mtd_restart.hills.q0s.shape[0]==6
            assert np.all(mtd_restart.hills.q0s[:3]==mtd.hills.q0s)
            assert 'hills' in f1
            assert np.all(f1['hills/q0'][:]==mtd_restart.hills.q0s)
            assert np.all(f1['hills/K'][:]==mtd_restart.hills.Ks)
            assert f1['hills/sigma'].shape[0]==1
            assert f1['hills/sigma'][0]==sigma
