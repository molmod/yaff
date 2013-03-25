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


def get_system_ne():
    return System(
        numbers=np.array([10]),
        pos=np.zeros((1, 3), float),
        ffatypes=['Ne'],
        rvecs=np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
    )


def test_grid_basics1():
    s = get_system_ne()
    grids = {'Ne': np.array([[[1.1]]])}

    fp = ForcePartGrid(s, grids)
    ff = ForceField(s, [fp])

    assert ff.compute() == 1.1


def test_grid_basics2():
    s = get_system_ne()

    grids = {'Ne': np.random.uniform(0, 1, (2, 2, 2))}

    fp = ForcePartGrid(s, grids)
    ff = ForceField(s, [fp])

    assert ff.compute() == grids['Ne'][0,0,0]
    ff.update_pos(np.array([[2.5, 2.5, 2.5]]))
    assert abs(ff.compute() - grids['Ne'].mean()) < 1e-12
    ff.update_pos(np.array([[7.5, 2.5, 7.5]]))
    assert abs(ff.compute() - grids['Ne'].mean()) < 1e-12
    ff.update_pos(np.array([[-2.5, 7.5, 2.5]]))
    assert abs(ff.compute() - grids['Ne'].mean()) < 1e-12
    ff.update_pos(np.array([[5.0, -5.0, -15.0]]))
    assert abs(ff.compute() - grids['Ne'][1,1,1]) < 1e-12
    ff.update_pos(np.array([[0.0, 20.0, 5.0]]))
    assert abs(ff.compute() - grids['Ne'][0,0,1]) < 1e-12


def check_continuity(ff, pos, eps, scale):
    ff.update_pos(pos)
    e0 = ff.compute()
    pos += np.random.uniform(-eps, eps, (1, 3))
    ff.update_pos(pos)
    e1 = ff.compute()
    assert abs(e1-e0) < eps*scale


def test_grid_continuity():
    s = get_system_ne()
    grids = {'Ne': np.random.uniform(0, 1, (20, 20, 20))}

    fp = ForcePartGrid(s, grids)
    ff = ForceField(s, [fp])

    eps = 1e-5
    for i in xrange(100):
        check_continuity(ff, np.random.uniform(-10, 20, (1, 3)), eps, 40)
    check_continuity(ff, np.array([[0.5, 1.234, 74.555]]), eps, 40)
    check_continuity(ff, np.array([[1.234, 0.5, 74.555]]), eps, 40)
    check_continuity(ff, np.array([[1.234, 74.555, 0.5]]), eps, 40)


def test_grid_conincide():
    s = get_system_ne()
    grids = {'Ne': np.random.uniform(0, 1, (5, 5, 5))}

    fp = ForcePartGrid(s, grids)
    ff = ForceField(s, [fp])

    for i in xrange(100):
        indexes = np.random.randint(-30, 50, 3)
        e0 = grids['Ne'][tuple(indexes%5)]
        pos = np.array([indexes*2.0])
        ff.update_pos(pos)
        e1 = ff.compute()
        assert abs(e0-e1) < 1e-10
