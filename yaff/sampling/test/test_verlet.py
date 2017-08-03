# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
#--


import pkg_resources
import h5py as h5
import numpy as np

from yaff import *
from yaff.test.common import get_system_water
from yaff.sampling.test.common import get_ff_water32, get_ff_water


def test_basic_water32():
    nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond)
    nve.run(5)
    assert nve.counter == 5


def test_basic_water():
    nve = VerletIntegrator(get_ff_water(), 1.0*femtosecond)
    nve.run(5)
    assert nve.counter == 5


def check_hdf5_common(f, isolated=False):
    assert 'system' in f
    assert 'numbers' in f['system']
    assert 'ffatypes' in f['system']
    assert 'ffatype_ids' in f['system']
    assert 'pos' in f['system']
    assert 'bonds' in f['system']
    assert ('rvecs' in f['system']) ^ isolated
    assert 'charges' in f['system']
    assert 'trajectory' in f
    assert 'counter' in f['trajectory']
    assert 'time' in f['trajectory']
    assert 'epot' in f['trajectory']
    assert 'pos' in f['trajectory']
    assert 'vel' in f['trajectory']
    assert 'rmsd_delta' in f['trajectory']
    assert 'rmsd_gpos' in f['trajectory']
    assert 'ekin' in f['trajectory']
    assert 'temp' in f['trajectory']
    assert 'etot' in f['trajectory']
    assert 'econs' in f['trajectory']
    assert 'dipole' in f['trajectory']
    assert 'dipole_vel' in f['trajectory']
    assert 'epot_contribs' in f['trajectory']
    assert 'epot_contrib_names' in f['trajectory'].attrs


def test_hdf5():
    with h5.File('yaff.sampling.test.test_verlet.test_hdf5.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=hdf5)
        nve.run(15)
        assert nve.counter == 15
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 16
        assert f['trajectory/counter'][15] == 15


def test_hdf5_start():
    with h5.File('yaff.sampling.test.test_verlet.test_hdf5_start.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f, start=2)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=hdf5)
        nve.run(5)
        assert nve.counter == 5
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 4
        assert f['trajectory/counter'][3] == 5


def test_hdf5_step():
    with h5.File('yaff.sampling.test.test_verlet.test_hdf5_step.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f, step=2)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=hdf5)
        nve.run(5)
        assert nve.counter == 5
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 3
        assert f['trajectory/counter'][2] == 4


def test_hdf5_simple():
    # This test does not write all possible outputs
    sys = get_system_water()
    ff = ForceField.generate(sys, pkg_resources.resource_filename(__name__, '../../data/test/parameters_water_bondharm.txt'))
    with h5.File('yaff.sampling.test.test_verlet.test_hdf5_simple.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f)
        nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=hdf5)
        nve.run(15)
        assert nve.counter == 15
        check_hdf5_common(hdf5.f, isolated=True)
        assert get_last_trajectory_row(f['trajectory']) == 16
        assert f['trajectory/counter'][15] == 15


def test_xyz():
    xyz = XYZWriter('/dev/null')
    nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=[xyz])
    com_vel = np.dot(nve.masses, nve.vel)/nve.masses.sum()
    nve.run(15)
    com_vel = np.dot(nve.masses, nve.vel)/nve.masses.sum()
    assert nve.counter == 15


def test_xyz_select():
    xyz = XYZWriter('/dev/null', select=[0,1,2])
    nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=[xyz])
    nve.run(15)
    assert nve.counter == 15


def test_kinetic_annealing():
    nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=KineticAnnealing())
    nve.run(5)
    assert nve.counter == 5
