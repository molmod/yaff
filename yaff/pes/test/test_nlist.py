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


import random
import numpy as np
from nose.plugins.skip import SkipTest

from molmod import angstrom

from yaff.test.common import get_system_water32, get_system_graphene8, \
    get_system_polyethylene4, get_system_quartz, get_system_glycine

from yaff import *


def test_nlist_water32_4A():
    system = get_system_water32()
    nlist = NeighborList(system)
    rcut = 4*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nneigh = nlist.nneigh
    assert (nlist.neighs['a'][:nneigh] > nlist.neighs['b'][:nneigh]).all()
    # check a few random rows from the neighbor list
    for i in random.sample(xrange(nneigh), 100):
        row = nlist.neighs[i]
        assert row['d'] <= rcut

        delta = system.pos[row['b']] - system.pos[row['a']]
        delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
        d = np.linalg.norm(delta)

        assert abs(d - row['d']) < 1e-8
        assert abs(delta[0] - row['dx']) < 1e-8
        assert abs(delta[1] - row['dy']) < 1e-8
        assert abs(delta[2] - row['dz']) < 1e-8
        assert row['r0'] == 0
        assert row['r1'] == 0
        assert row['r2'] == 0


def test_nlist_water32_9A():
    system = get_system_water32()
    nlist = NeighborList(system)
    rcut = 9*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nneigh = nlist.nneigh
    assert (
        (nlist.neighs['a'][:nneigh] > nlist.neighs['b'][:nneigh]) |
        (nlist.neighs['r0'][:nneigh] != 0) |
        (nlist.neighs['r1'][:nneigh] != 0) |
        (nlist.neighs['r2'][:nneigh] != 0)
    ).all()
    for a in random.sample(xrange(system.natom), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for b in xrange(system.natom):
            delta = system.pos[b] - system.pos[a]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            assert abs(delta).max() < 0.5*9.865*angstrom
            for l2 in xrange(0, 2):
                for l1 in xrange((l2!=0)*-1, 2):
                    for l0 in xrange((l2!=0 or l1!=0)*-1, 2):
                        my_delta = delta + np.array([l0,l1,l2])*9.865*angstrom
                        d = np.linalg.norm(my_delta)
                        if d <= rcut:
                            if (l0!=0) or (l1!=0) or (l2!=0) or (a>b):
                                check[(b, l0, l1, l2)] = (d, my_delta)
        # compare
        counter = 0
        for row in nlist.neighs[:nneigh]:
            if row['a'] == a:
                key = row['b'], row['r0'], row['r1'], row['r2']
                assert key in check
                assert abs(check[key][0]) <= rcut
                assert abs(check[key][0] - row['d']) < 1e-8
                assert abs(check[key][1][0] - row['dx']) < 1e-8
                assert abs(check[key][1][1] - row['dy']) < 1e-8
                assert abs(check[key][1][2] - row['dz']) < 1e-8
                counter += 1
        assert counter == len(check)


def test_nlist_graphene8_9A():
    system = get_system_graphene8()
    nlist = NeighborList(system)
    rcut = 9*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nlist.check()


def test_nlist_polyethylene4_9A():
    system = get_system_polyethylene4()
    nlist = NeighborList(system)
    rcut = 9*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nlist.check()


def test_nlist_quartz_4A_shortest():
    raise SkipTest('The mic routine fails to find the shortest distance in small skewed unit cells.')
    system = get_system_quartz()
    nlist = NeighborList(system)
    nlist.request_rcut(4*angstrom)
    nlist.update()
    check_nlist_shortest(system, nlist)


def test_nlist_water32_9A_shortest():
    system = get_system_water32()
    nlist = NeighborList(system)
    nlist.request_rcut(9*angstrom)
    nlist.update()
    check_nlist_shortest(system, nlist)


def check_nlist_shortest(system, nlist):
    for row in nlist.neighs[:nlist.nneigh]:
        if (row['r0'] == 0) and (row['r1'] == 0) and (row['r2'] == 0):
            delta0 = np.array([row['dx'], row['dy'], row['dz']])
            delta1 = delta0.copy()
            system.cell.mic(delta1)
            assert abs(delta0 - delta1).max() < 1e-10
            for r0 in xrange(-1, 1):
                for r1 in xrange(-1, 1):
                    for r2 in xrange(-1, 1):
                        if (r0==0) and (r1==0) and (r2==0):
                            continue
                        delta = delta0 + r0*system.cell.rvecs[0] + r1*system.cell.rvecs[1] + r2*system.cell.rvecs[2]
                        assert np.linalg.norm(delta) >= row['d']


def test_nlist_quartz_9A():
    system = get_system_quartz()
    nlist = NeighborList(system)
    rcut = 9*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nlist.check()


def test_nlist_quartz_20A():
    system = get_system_quartz()
    nlist = NeighborList(system)
    rcut = 20*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nlist.check()


def test_nlist_quartz_110A():
    system = get_system_quartz()
    nlist = NeighborList(system)
    rcut = 110*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    assert (nlist.neighs['r0'][:nlist.nneigh] <= nlist.rmax[0]).all()
    assert (nlist.neighs['r0'][:nlist.nneigh] >= -nlist.rmax[0]).all()
    assert (nlist.neighs['r1'][:nlist.nneigh] <= nlist.rmax[1]).all()
    assert (nlist.neighs['r1'][:nlist.nneigh] >= -nlist.rmax[1]).all()
    assert (nlist.neighs['r2'][:nlist.nneigh] <= nlist.rmax[2]).all()
    assert (nlist.neighs['r2'][:nlist.nneigh] >= 0).all()


def test_nlist_glycine_9A():
    system = get_system_glycine()
    nlist = NeighborList(system)
    rcut = 9*angstrom
    nlist.request_rcut(rcut)
    nlist.update()
    nlist.check()


def test_nlist_inc_r3():
    cell = get_system_water32().cell
    rmax = np.array([2, 2, 2])
    r = np.array([-2, 1, 1])
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-1, 1, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([0, 1, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([1, 1, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([2, 1, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-2, 2, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-1, 2, 1])).all()
    r = np.array([2, 2, 0])
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-2, -2, 1])).all()
    r = np.array([2, 2, 2])
    assert not nlist_inc_r(cell, r, rmax)
    assert (r == np.array([0, 0, 0])).all()


def test_nlist_inc_r2():
    cell = get_system_graphene8().cell
    rmax = np.array([2, 2])
    r = np.array([-2, 1])
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-1, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([0, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([1, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([2, 1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-2, 2])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([-1, 2])).all()
    r = np.array([2, 2])
    assert not nlist_inc_r(cell, r, rmax)
    assert (r == np.array([0, 0])).all()


def test_nlist_inc_r1():
    cell = get_system_polyethylene4().cell
    rmax = np.array([2])
    r = np.array([0])
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([1])).all()
    assert nlist_inc_r(cell, r, rmax)
    assert (r == np.array([2])).all()
    assert not nlist_inc_r(cell, r, rmax)
    assert (r == np.array([0])).all()


def test_nlist_inc_r0():
    cell = get_system_glycine().cell
    rmax = np.array([], dtype=int)
    r = np.array([], dtype=int)
    assert not nlist_inc_r(cell, r, rmax)
    assert (r == np.array([], dtype=int)).all()


def check_nlist_skin(system, rcut, skin):
    nlist1 = NeighborList(system, skin)
    nlist1.request_rcut(rcut)
    nlist1.update()
    # Displace all atoms with a random vector the rebuild is not triggered
    for i in xrange(system.natom):
        vec = np.random.normal(-1, 1, 3)
        vec *= 0.45/(nlist1.rmax.max()+1)*skin/np.linalg.norm(vec)
        system.pos[i] += vec
    assert not nlist1._need_rebuild()
    nlist1.update()

    nlist2 = NeighborList(system)
    nlist2.request_rcut(rcut)

    # Check if all distances present in nlist2 are also present in nlist1.
    lookup = {}
    for row in nlist1.neighs[:nlist1.nneigh]:
        key = row['a'], row['b'], row['r0'], row['r1'], row['r2']
        value = row['d'], row['dx'], row['dy'], row['dz']
        lookup[key] = value

    for row in nlist2.neighs[:nlist2.nneigh]:
        key = row['a'], row['b'], row['r0'], row['r1'], row['r2']
        value = lookup.get(key)
        assert value is not None
        d, dx, dy, dz = value

        assert abs(d - row['d']) < 1e-8
        assert abs(dx - row['dx']) < 1e-8
        assert abs(dy - row['dy']) < 1e-8
        assert abs(dz - row['dz']) < 1e-8

    # Displace all atoms with a random vector the rebuild is triggered.
    for i in xrange(system.natom):
        vec = np.random.normal(-1, 1, 3)
        vec *= 0.55/(nlist1.rmax.max()+1)*skin/np.linalg.norm(vec)
        system.pos[i] += vec
    assert nlist1._need_rebuild()


def test_nlist_quartz_6A_skin3A():
    system = get_system_quartz()
    check_nlist_skin(system, 6*angstrom, 3*angstrom)


def test_nlist_water32_6A_skin3A():
    system = get_system_water32()
    check_nlist_skin(system, 6*angstrom, 3*angstrom)


def test_nlist_water32_10A_skin2A():
    system = get_system_water32()
    check_nlist_skin(system, 10*angstrom, 2*angstrom)
