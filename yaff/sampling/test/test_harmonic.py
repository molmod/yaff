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


import numpy as np

from yaff import *
from yaff.sampling.test.common import get_ff_water32, get_ff_water, get_ff_bks


def test_hessian_partial_water32():
    ff = get_ff_water32()
    select = [1, 2, 3, 14, 15, 16]
    hessian = estimate_cart_hessian(ff, select=select)
    assert hessian.shape == (18, 18)


def test_hessian_full_water():
    ff = get_ff_water()
    hessian = estimate_cart_hessian(ff)
    assert hessian.shape == (9, 9)
    evals = np.linalg.eigvalsh(hessian)
    print evals
    assert sum(abs(evals) < 1e-10) == 3


def test_hessian_full_x2():
    K, d = np.random.uniform(1.0, 2.0, 2)
    system = System(
        numbers=np.array([1, 1]),
        pos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]]),
        ffatypes=['H', 'H'],
        bonds=np.array([[0, 1]]),
    )
    part = ForcePartValence(system)
    part.add_term(Harmonic(K, d, Bond(0, 1)))
    ff = ForceField(system, [part])
    hessian = estimate_cart_hessian(ff)
    evals = np.linalg.eigvalsh(hessian)
    assert abs(evals[:-1]).max() < 1e-5
    assert abs(evals[-1] - 2*K) < 1e-5


def test_elastic_water32():
    ff = get_ff_water32()
    elastic = estimate_elastic(ff, do_frozen=True)
    assert elastic.shape == (6, 6)


def test_bulk_elastic_bks():
    ff = get_ff_bks(smooth_ei=True, reci_ei='ignore')
    system = ff.system
    lcs = np.array([
        [1, 1, 0],
        [0, 0, 1],
    ])
    system.align_cell(lcs)
    ff.update_rvecs(system.cell.rvecs)
    opt = QNOptimizer(FullCellDOF(ff, gpos_rms=1e-6, grvecs_rms=1e-6))
    opt.run()
    rvecs0 = system.cell.rvecs.copy()
    vol0 = system.cell.volume
    pos0 = system.pos.copy()
    e0 = ff.compute()
    elastic = estimate_elastic(ff)
    assert abs(pos0 - system.pos).max() < 1e-10
    assert abs(rvecs0 - system.cell.rvecs).max() < 1e-10
    assert abs(vol0 - system.cell.volume) < 1e-10
    assert elastic.shape == (6, 6)
    # Make estimates of the same matrix elements with a simplistic approach
    eps = 1e-3

    from nose.plugins.skip import SkipTest
    raise SkipTest('Double check elastic constant implementation')

    # A) stretch in the Z direction
    deform = np.array([1, 1, 1-eps])
    rvecs1 = rvecs0*deform
    pos1 = pos0*deform
    ff.update_rvecs(rvecs1)
    ff.update_pos(pos1)
    opt = QNOptimizer(CartesianDOF(ff, gpos_rms=1e-6))
    opt.run()
    e1 = ff.compute()
    deform = np.array([1, 1, 1+eps])
    rvecs2 = rvecs0*deform
    pos2 = pos0*deform
    ff.update_rvecs(rvecs2)
    ff.update_pos(pos2)
    opt = QNOptimizer(CartesianDOF(ff, gpos_rms=1e-6))
    opt.run()
    e2 = ff.compute()
    C = (e1 + e2 - 2*e0)/(eps**2)/vol0
    assert abs(C - elastic[2,2]) < C*0.02

    # B) stretch in the X direction
    deform = np.array([1-eps, 1, 1])
    rvecs1 = rvecs0*deform
    pos1 = pos0*deform
    ff.update_rvecs(rvecs1)
    ff.update_pos(pos1)
    opt = QNOptimizer(CartesianDOF(ff, gpos_rms=1e-6))
    opt.run()
    e1 = ff.compute()
    deform = np.array([1+eps, 1, 1])
    rvecs2 = rvecs0*deform
    pos2 = pos0*deform
    ff.update_rvecs(rvecs2)
    ff.update_pos(pos2)
    opt = QNOptimizer(CartesianDOF(ff, gpos_rms=1e-6))
    opt.run()
    e2 = ff.compute()
    C = (e1 + e2 - 2*e0)/(eps**2)/vol0
    assert abs(C - elastic[0,0]) < C*0.02
