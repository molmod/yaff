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

from nose.tools import assert_raises
import numpy as np

from yaff.test.common import get_system_water32, get_system_glycine, \
    get_system_quartz, get_system_caffeine, get_system_mil53

from yaff import *


def test_scaling_water32():
    system = get_system_water32()
    stab = Scalings(system, 0.5, 0.0, 1.0).stab
    assert (stab['a'] > stab['b']).all()
    assert len(stab) == system.natom
    for i0, i1, scale, nbond in stab:
        if system.numbers[i1] == 8:
            assert (i0 == i1+1) or (i0 == i1+2)
            assert scale == 0.5
            assert nbond == 1
        elif system.numbers[i1] == 1:
            assert i0 == i1+1
            assert scale == 0.0
            assert nbond == 2


def test_scaling_glycine():
    system = get_system_glycine()
    stab = Scalings(system, 1.0, 0.5, 0.2).stab # warning: absurd numbers
    assert (stab['a'] > stab['b']).all()
    assert len(stab) == sum(len(system.neighs2[i]) + len(system.neighs3[i]) for i in range(system.natom))//2
    for i0, i1, scale, nbond in stab:
        if i0 in system.neighs2[i1]:
            assert scale == 0.5
            assert nbond == 2
        elif i0 in system.neighs3[i1]:
            assert scale == 0.2
            assert nbond == 3


def test_scaling_quartz():
    system = get_system_quartz().supercell(2, 2, 2)
    stab = Scalings(system).stab
    assert (stab['a'] > stab['b']).all()
    assert len(stab) == sum(len(system.neighs1[i]) + len(system.neighs2[i]) for i in range(system.natom))//2
    for i0, i1, scale, nbond in stab:
        assert scale == 0.0
        assert i0 in system.neighs1[i1] or i0 in system.neighs2[i1]
        assert nbond == 1 or nbond == 2


def test_iter_paths1():
    system = get_system_caffeine()
    paths = set(iter_paths(system, 2, 8, 3))
    assert all(len(path) == 4 for path in paths)
    assert len(paths) == 2
    assert paths == set([(2, 7, 6, 8), (2, 9, 4, 8)])


def test_iter_paths2():
    system = get_system_caffeine()
    paths = set(iter_paths(system, 13, 5, 5))
    assert all(len(path) == 6 for path in paths)
    assert len(paths) == 2
    assert paths == set([(13, 4, 8, 6, 7, 5), (13, 4, 9, 2, 7, 5)])


def test_iter_paths3():
    system = get_system_caffeine()
    paths = set(iter_paths(system, 18, 19, 2))
    assert all(len(path) == 3 for path in paths)
    assert len(paths) == 1
    assert paths == set([(18, 12, 19)])


def test_scaling_mil53():
    system = get_system_mil53()
    with assert_raises(AssertionError):
        scalings = Scalings(system)
