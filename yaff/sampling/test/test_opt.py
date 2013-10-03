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
from yaff.sampling.test.common import get_ff_water32, get_ff_bks
from yaff.pes.test.common import check_gpos_part, check_vtens_part, \
    check_gpos_ff, check_vtens_ff


def test_cg_5steps():
    dof = CartesianDOF(get_ff_water32())
    dof.check_delta()
    opt = CGOptimizer(dof)
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_cg_5steps_partial():
    dof = CartesianDOF(get_ff_water32(), select=[0, 1, 2, 3, 4, 5])
    dof.check_delta()
    opt = CGOptimizer(dof)
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_cg_full_cell_5steps():
    dof = FullCellDOF(get_ff_water32())
    dof.check_delta()
    opt = CGOptimizer(dof)
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_cg_aniso_cell_5steps():
    dof = AnisoCellDOF(get_ff_water32())
    dof.check_delta()
    opt = CGOptimizer(dof)
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_cg_iso_cell_5steps():
    dof = IsoCellDOF(get_ff_water32())
    dof.check_delta
    opt = CGOptimizer(dof)
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_cg_until_converged():
    opt = CGOptimizer(CartesianDOF(get_ff_water32(), gpos_rms=1e-1, dpos_rms=None))
    assert opt.dof.th_gpos_rms == 1e-1
    assert opt.dof.th_dpos_rms is None
    opt.run()
    assert opt.dof.conv_count == 0
    assert opt.dof.conv_val < 1
    assert opt.dof.conv_worst.startswith('gpos_')
    assert opt.dof.gpos_max < 1e-1*3
    assert opt.dof.gpos_rms < 1e-1


def check_hdf5_common(f):
    assert 'system' in f
    assert 'numbers' in f['system']
    assert 'ffatypes' in f['system']
    assert 'ffatype_ids' in f['system']
    assert 'pos' in f['system']
    assert 'bonds' in f['system']
    assert 'rvecs' in f['system']
    assert 'charges' in f['system']
    assert 'trajectory' in f
    assert 'counter' in f['trajectory']
    assert 'epot' in f['trajectory']
    assert 'pos' in f['trajectory']
    assert 'dipole' in f['trajectory']
    assert 'epot_contribs' in f['trajectory']
    assert 'epot_contrib_names' in f['trajectory'].attrs


def test_cg_hdf5():
    f = h5.File('yaff.sampling.test.test_opt.test_cg_hdf5.h5', driver='core', backing_store=False)
    try:
        hdf5 = HDF5Writer(f)
        opt = CGOptimizer(CartesianDOF(get_ff_water32()), hooks=hdf5)
        opt.run(15)
        assert opt.counter == 15
        check_hdf5_common(hdf5.f)
        assert get_last_trajectory_row(f['trajectory']) == 16
        assert f['trajectory/counter'][15] == 15
    finally:
        f.close()


def test_qn_5steps():
    opt = QNOptimizer(CartesianDOF(get_ff_water32()))
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_qn_5steps_initial_hessian():
    ff = get_ff_water32()
    hessian = estimate_cart_hessian(ff)
    opt = QNOptimizer(CartesianDOF(ff), hessian0=hessian)
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_solve_trust_radius_random1():
    N = 10
    eps = 1e-4
    for i in xrange(100):
        grad = np.random.normal(0, 1, N)
        evals = np.random.normal(0, 1, N)
        step = solve_trust_radius(grad, evals, 1, eps)
        assert np.linalg.norm(step) <= 1.0+eps


def test_solve_trust_radius_random2():
    N = 10
    eps = 1e-4
    for i in xrange(100):
        grad = np.random.normal(0, 1, N)
        evals = np.exp(np.random.normal(0, 3, N))
        step = solve_trust_radius(grad, evals, 1, eps)
        assert np.linalg.norm(step) <= 1.0+eps


def test_solve_trust_radius_random3():
    N = 10
    eps = 1e-4
    for i in xrange(100):
        grad = np.random.normal(0, 1, N)
        evals = np.exp(np.random.normal(0, 3, N))-0.01
        step = solve_trust_radius(grad, evals, 1, eps)
        assert np.linalg.norm(step) <= 1.0+eps
