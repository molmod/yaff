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

from yaff import DeltaList

from yaff.test.common import get_system_graphene8, get_system_quartz, \
    get_system_glycine


def get_dlist_bonds(system):
    dlist = DeltaList(system)
    for i, j in system.bonds:
        dlist.add_delta(i, j)
    return dlist


def get_dlist_random(system, n=10):
    dlist = DeltaList(system)
    for row in xrange(n):
        i = np.random.randint(system.natom)
        j = (i + np.random.randint(1, system.natom)) % system.natom
        dlist.add_delta(i, j)
    return dlist


def check_dlist(system, dlist):
    dlist.forward()
    for row in dlist.deltas[:dlist.ndelta]:
        i = row['i']
        j = row['j']
        delta = system.pos[j] - system.pos[i]
        system.cell.mic(delta)
        assert abs(delta[0] - row['dx']) < 1e-5
        assert abs(delta[1] - row['dy']) < 1e-5
        assert abs(delta[2] - row['dz']) < 1e-5


def test_dlist_graphene8_bonds():
    system = get_system_graphene8()
    dlist = get_dlist_bonds(system)
    check_dlist(system, dlist)


def test_dlist_graphene8_random():
    system = get_system_graphene8()
    dlist = get_dlist_random(system)
    check_dlist(system, dlist)


def test_dlist_quartz_bonds():
    system = get_system_quartz()
    dlist = get_dlist_bonds(system)
    check_dlist(system, dlist)


def test_dlist_quartz_random():
    system = get_system_quartz()
    dlist = get_dlist_random(system)
    check_dlist(system, dlist)


def test_dlist_glycine_bonds():
    system = get_system_glycine()
    dlist = get_dlist_bonds(system)
    check_dlist(system, dlist)


def test_dlist_glycine_random():
    system = get_system_glycine()
    dlist = get_dlist_random(system, 100)
    assert dlist.ndelta <= 45
    check_dlist(system, dlist)
