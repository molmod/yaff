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

import numpy as np

from molmod.minimizer import check_delta

from yaff import *


def test_hammer_deriv():
    rcut = 10.0
    for tau in 0.5, 1.0, 2.0:
        tr = Hammer(tau)

        def fn(x, do_gradient):
            value, gradient = tr.trunc_fn(x[0], rcut)
            if do_gradient:
                return value, np.array([gradient])
            else:
                return value

        for d in 5.0, 9.0, 9.9:
            x0 = np.array([d])
            dxs = np.random.normal(0, 1e-4, 100)
            check_delta(fn, x0, dxs)


def test_hammer():
    rcut = 10.0

    tr = Hammer(1.0)
    assert tr.trunc_fn(9.99, rcut)[0] < 1e-10
    assert tr.trunc_fn(9.99, rcut)[1] < 1e-10
    assert tr.trunc_fn(10.0, rcut) == (0.0, 0.0)
    assert tr.trunc_fn(0.0, rcut)[0] > 0.9
    assert tr.trunc_fn(0.0, rcut)[1] <= 0.0
    assert abs(tr.trunc_fn(0.0, rcut)[1]) < 1e-2

    tr = Hammer(0.5)
    assert tr.trunc_fn(9.99, rcut)[0] < 1e-10
    assert tr.trunc_fn(9.99, rcut)[1] < 1e-10
    assert tr.trunc_fn(10.0, rcut) == (0.0, 0.0)
    assert tr.trunc_fn(0.0, rcut)[0] > 0.95
    assert tr.trunc_fn(0.0, rcut)[1] <= 0.0
    assert abs(tr.trunc_fn(0.0, rcut)[1]) < 5e-3


def test_switch3_deriv():
    rcut = 10.0
    for width in 1.0, 2.0, 4.0:
        tr = Switch3(width)

        def fn(x, do_gradient):
            value, gradient = tr.trunc_fn(x[0], rcut)
            if do_gradient:
                return value, np.array([gradient])
            else:
                return value

        for d in 5.0, 9.5:
            x0 = np.array([d])
            dxs = np.random.normal(0, 1e-4, 100)
            check_delta(fn, x0, dxs)


def test_switch3():
    rcut = 10.0

    tr = Switch3(1.0)
    assert tr.trunc_fn(9.99, rcut)[0] < 3e-3
    assert tr.trunc_fn(9.99, rcut)[1] < 6e-2
    assert tr.trunc_fn(10.0, rcut) == (0.0, 0.0)
    assert tr.trunc_fn(0.0, rcut) == (1.0, 0.0)
    assert tr.trunc_fn(9.0, rcut) == (1.0, 0.0)
    assert tr.trunc_fn(8.0, rcut) == (1.0, 0.0)
    assert tr.trunc_fn(9.5, rcut) == (0.5, -1.5)

    tr = Switch3(0.5)
    assert tr.trunc_fn(9.99, rcut)[0] < 2e-3
    assert tr.trunc_fn(9.99, rcut)[1] < 0.2
    assert tr.trunc_fn(10.0, rcut) == (0.0, 0.0)
    assert tr.trunc_fn(0.0, rcut) == (1.0, 0.0)
    assert tr.trunc_fn(9.5, rcut) == (1.0, 0.0)
    assert tr.trunc_fn(8.0, rcut) == (1.0, 0.0)
