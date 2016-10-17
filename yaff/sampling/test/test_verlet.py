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


import h5py as h5, numpy as np

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
    f = h5.File('yaff.sampling.test.test_verlet.test_hdf5.h5', driver='core', backing_store=False)
    try:
        hdf5 = HDF5Writer(f)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=hdf5)
        nve.run(15)
        assert nve.counter == 15
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 16
        assert f['trajectory/counter'][15] == 15
    finally:
        f.close()


def test_hdf5_start():
    f = h5.File('yaff.sampling.test.test_verlet.test_hdf5_start.h5', driver='core', backing_store=False)
    try:
        hdf5 = HDF5Writer(f, start=2)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=hdf5)
        nve.run(5)
        assert nve.counter == 5
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 4
        assert f['trajectory/counter'][3] == 5
    finally:
        f.close()


def test_hdf5_step():
    f = h5.File('yaff.sampling.test.test_verlet.test_hdf5_step.h5', driver='core', backing_store=False)
    try:
        hdf5 = HDF5Writer(f, step=2)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=hdf5)
        nve.run(5)
        assert nve.counter == 5
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 3
        assert f['trajectory/counter'][2] == 4
    finally:
        f.close()


def test_hdf5_simple():
    # This test does not write all possible outputs
    sys = get_system_water()
    ff = ForceField.generate(sys, context.get_fn('test/parameters_water_bondharm.txt'))
    f = h5.File('yaff.sampling.test.test_verlet.test_hdf5_simple.h5', driver='core', backing_store=False)
    try:
        hdf5 = HDF5Writer(f)
        nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=hdf5)
        nve.run(15)
        assert nve.counter == 15
        check_hdf5_common(hdf5.f, isolated=True)
        assert get_last_trajectory_row(f['trajectory']) == 16
        assert f['trajectory/counter'][15] == 15
    finally:
        f.close()


def test_hdf5_restart():
    # Basic test for RestartWriter; original coder of that Class should write
    # unit tests! Yes, Sven Rogge that will be you!
    f0 = h5.File('yaff.sampling.test.test_verlet.test_hdf5_restart0.h5', driver='core', backing_store=False)
    f1 = h5.File('yaff.sampling.test.test_verlet.test_hdf5_restart1.h5', driver='core', backing_store=False)
    f2 = h5.File('yaff.sampling.test.test_verlet.test_hdf5_restart2.h5', driver='core', backing_store=False)
    np.random.seed(3)
    try:
        hdf5 = HDF5Writer(f0)
        # Write a checkpoint every three steps
        restart = RestartWriter(f1, step=3)
        # Run reference simulation for 5 steps
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=[hdf5,restart])
        nve.run(5)
        # Run restart simulation for 2 additional steps, starting from restart after three steps
        hdf5_restart = HDF5Writer(f2)
        nver = VerletIntegrator(get_ff_water32(), restart_h5=f1, hooks=[hdf5_restart])
        nver.run(2)
        # Check that resumed simulation gives same results as original
        assert nve.counter == 5
        assert nver.counter == 5
        print f2['trajectory/econs'][:]
        print f0['trajectory/econs'][:]
        print f0['trajectory'].keys()
        nsteps = f2['trajectory/counter'][:].shape[0]
        for item in 'counter','pos','epot','econs':
            assert np.all( f0['trajectory/%s'%item][-nsteps:]==f2['trajectory/%s'%item][:] )
        assert False
    finally:
        f0.close()
        f1.close()
        f2.close()


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
