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

from yaff import *
from yaff.test.common import get_system_water
from yaff.sampling.test.common import get_ff_water32, get_ff_water
from molmod.test.common import tmpdir


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


def test_hdf5_cvs():
    # This test checks that CVStateItem and BiasStateItem writes output
    ff = get_ff_water32()
    part_bias = ForcePartBias(ff.system)
    ff.add_part(part_bias)
    cv0 = CVVolume(ff.system)
    K0, V0 = 0.1, 0.8*ff.system.cell.volume
    bias0 = HarmonicBias(K0,V0,cv0)
    part_bias.add_term(bias0)
    cv1 = Bond(ff.system.bonds[0,0], ff.system.bonds[0,1])
    K1, r1 = 0.2, 0.8
    bias1 = Harmonic(K1, r1,cv1)
    part_bias.add_term(bias1)
    cv_tracker = CVStateItem([cv0,cv1])
    bias_tracker = BiasStateItem(part_bias)
    with h5.File('yaff.sampling.test.test_verlet.test_hdf5_cvs.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f)
        nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=hdf5, state=[cv_tracker, bias_tracker])
        nve.run(5)
        assert nve.counter == 5
        check_hdf5_common(hdf5.f, isolated=False)
        assert get_last_trajectory_row(f['trajectory']) == 6
        assert f['trajectory/counter'][5] == 5
        cv_values = f['trajectory/cv_values'][:]
        assert cv_values.shape[0]==6
        assert cv_values.shape[1]==2
        assert np.all(cv_values[:,0]==f['trajectory/volume'][:])
        ref_bond_lengths = []
        for pos in f['trajectory/pos'][:]:
            delta = pos[ff.system.bonds[0,0]] - pos[ff.system.bonds[0,1]]
            ff.system.cell.mic(delta)
            ref_bond_lengths.append(np.linalg.norm(delta))
        ref_bond_lengths = np.asarray(ref_bond_lengths)
        assert np.all(np.abs(cv_values[:,1]-ref_bond_lengths)<1e-5)
        bias_values = f['trajectory/bias_values'][:]
        assert bias_values.shape[0]==6
        assert bias_values.shape[1]==2
        assert np.all(np.abs(0.5*K0*(cv_values[:,0]-V0)**2 - bias_values[:,0])<1e-5)
        assert np.all(np.abs(0.5*K1*(cv_values[:,1]-r1)**2 - bias_values[:,1])<1e-5)
        assert 'cv_names' in f['trajectory'].attrs


def test_xyz():
    with tmpdir(__name__, 'test_xyz') as dn:
        fn_xyz = os.path.join(dn, 'foobar.xyz')
        xyz = XYZWriter(fn_xyz)
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=[xyz])
        com_vel = np.dot(nve.masses, nve.vel)/nve.masses.sum()
        nve.run(15)
        com_vel = np.dot(nve.masses, nve.vel)/nve.masses.sum()
        assert os.path.isfile(fn_xyz)
        assert nve.counter == 15
        ## Ugly hack to make tests pass on Windows. The root cause is that the SliceReader
        ## in molmod.io.common is poorly written.
        xyz.xyz_writer._auto_close = False
        xyz.xyz_writer._f.close()


def test_xyz_select():
    with tmpdir(__name__, 'test_xyz_select') as dn:
        fn_xyz = os.path.join(dn, 'foobar.xyz')
        xyz = XYZWriter(fn_xyz, select=[0,1,2])
        nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=[xyz])
        nve.run(15)
        assert os.path.isfile(fn_xyz)
        assert nve.counter == 15
        ## Ugly hack to make tests pass on Windows. The root cause is that the SliceReader
        ## in molmod.io.common is poorly written.
        xyz.xyz_writer._auto_close = False
        xyz.xyz_writer._f.close()


def test_kinetic_annealing():
    nve = VerletIntegrator(get_ff_water32(), 1.0*femtosecond, hooks=KineticAnnealing())
    nve.run(5)
    assert nve.counter == 5
