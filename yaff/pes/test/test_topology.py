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

from yaff.test.common import get_system_water32, get_system_graphene8, \
    get_system_polyethylene4, get_system_quartz, get_system_glycine, \
    get_system_cyclopropene, get_system_caffeine, get_system_butanol

from yaff import *


def test_topology_water32():
    system = get_system_water32()
    assert system.bonds[0,0] == 0
    assert system.bonds[0,1] == 1
    assert system.bonds[1,0] == 0
    assert system.bonds[1,1] == 2
    assert system.bonds[2,0] == 3
    assert system.bonds[2,1] == 4
    assert system.bonds[3,0] == 3
    assert system.bonds[3,1] == 5
    for i in xrange(system.natom):
        if system.numbers[i] == 8:
            assert len(system.neighs1[i]) == 2
            n0, n1 = system.neighs1[i]
            assert system.numbers[n0] == 1
            assert system.numbers[n1] == 1
            assert len(system.neighs2[i]) == 0
            assert len(system.neighs3[i]) == 0
        elif system.numbers[i] == 1:
            assert len(system.neighs1[i]) == 1
            n, = system.neighs1[i]
            assert system.numbers[n] == 8
            assert len(system.neighs2[i]) == 1
            n, = system.neighs2[i]
            assert system.numbers[n] == 1
            assert len(system.neighs3[i]) == 0



def floyd_warshall(bonds, natom):
    '''A slow implementation of the Floyd-Warshall algorithm.

       Use it for small test systems only.
    '''
    dmat = np.zeros((natom, natom), int)+natom**2
    for i in xrange(natom):
        dmat[i,i] = 0
    for i0, i1 in bonds:
        dmat[i0,i1] = 1
        dmat[i1,i0] = 1
    for i0 in xrange(natom):
        for i1 in xrange(natom):
            for i2 in xrange(natom):
                if i2 == i1:
                    continue
                dmat[i1,i2] = min(dmat[i1,i2], dmat[i1,i0]+dmat[i0,i2])
    assert (dmat == dmat.transpose()).all()
    return dmat


def check_topology_slow(system):
    dmat = floyd_warshall(system.bonds, system.natom)
    # check dmat with neigs*
    for i0, n0 in system.neighs1.iteritems():
        for i1 in n0:
            assert dmat[i0, i1] == 1
            assert dmat[i1, i0] == 1
    for i0, n0 in system.neighs2.iteritems():
        for i2 in n0:
            assert dmat[i0, i2] == 2
            assert dmat[i2, i0] == 2
    for i0, n0 in system.neighs3.iteritems():
        for i3 in n0:
            assert dmat[i0, i3] == 3
            assert dmat[i3, i0] == 3
    # check neigs* with dmat
    for i0 in xrange(system.natom):
        for i1 in xrange(system.natom):
            if dmat[i0, i1] == 1:
                assert i1 in system.neighs1[i0]
            if dmat[i0, i1] == 2:
                assert i1 in system.neighs2[i0]
            if dmat[i0, i1] == 3:
                assert i1 in system.neighs3[i0]


def test_topology_graphene8():
    system = get_system_graphene8()
    check_topology_slow(system)


def test_topology_polyethylene4():
    system = get_system_polyethylene4()
    check_topology_slow(system)


def test_topology_quartz():
    system = get_system_quartz()
    check_topology_slow(system)


def test_topology_glycine():
    system = get_system_glycine()
    check_topology_slow(system)


def test_topology_cyclopropene():
    system = get_system_cyclopropene()
    check_topology_slow(system)


def test_topology_caffeine():
    system = get_system_caffeine()
    check_topology_slow(system)


def test_topology_butanol():
    system = get_system_butanol()
    check_topology_slow(system)
