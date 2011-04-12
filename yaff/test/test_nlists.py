# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
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


import random
import numpy as np

from molmod import angstrom
from common import get_system_water32, get_system_graphene8, \
    get_system_polyethylene4, get_system_quartz, get_system_glycine

from yaff import *


def test_nlists_water32_4A():
    system = get_system_water32()
    nlists = NeighborLists(system)
    cutoff = 4*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    assert len(nlists) == system.natom
    for i in random.sample(xrange(system.natom), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(i+1, system.natom):
            delta = system.pos[i] - system.pos[j]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            d = np.linalg.norm(delta)
            if d <= cutoff:
                check[j] = (d, delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            key = row['i']
            assert key in check
            assert abs(check[key][0]) <= cutoff
            assert abs(check[key][0] - row['d']) < 1e-8
            assert abs(check[key][1][0] - row['dx']) < 1e-8
            assert abs(check[key][1][1] - row['dy']) < 1e-8
            assert abs(check[key][1][2] - row['dz']) < 1e-8
            assert row['r0'] == 0
            assert row['r1'] == 0
            assert row['r2'] == 0


def test_nlists_water32_9A():
    system = get_system_water32()
    nlists = NeighborLists(system)
    cutoff = 9*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in random.sample(xrange(system.natom), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(0, system.natom):
            delta = system.pos[i] - system.pos[j]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            assert abs(delta).max() < 0.5*9.865*angstrom
            for l0 in xrange(-1, 2):
                for l1 in xrange(-1, 2):
                    for l2 in xrange(-1, 2):
                        my_delta = delta + np.array([l0,l1,l2])*9.865*angstrom
                        d = np.linalg.norm(my_delta)
                        if d <= cutoff:
                            if (l0!=0) or (l1!=0) or (l2!=0) or (j>i):
                                check[(j, l0, l1, l2)] = (d, my_delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            key = row['i'], row['r0'], row['r1'], row['r2']
            assert key in check
            assert abs(check[key][0]) <= cutoff
            assert abs(check[key][0] - row['d']) < 1e-8
            assert abs(check[key][1][0] - row['dx']) < 1e-8
            assert abs(check[key][1][1] - row['dy']) < 1e-8
            assert abs(check[key][1][2] - row['dz']) < 1e-8


def test_nlists_graphene8_9A():
    system = get_system_graphene8()
    nlists = NeighborLists(system)
    cutoff = 9*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in xrange(system.natom):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(0, system.natom):
            delta = system.pos[i] - system.pos[j]
            for c in xrange(len(system.rvecs)):
                delta -= system.rvecs[c]*np.ceil(np.dot(delta, system.gvecs[c]) - 0.5)
            for r0 in xrange(-3, 4):
                for r1 in xrange(-3, 4):
                    my_delta = delta + r0*system.rvecs[0] + r1*system.rvecs[1]
                    d = np.linalg.norm(my_delta)
                    if d <= cutoff:
                        if (r0!=0) or (r1!=0) or (j>i):
                            check[(j, r0, r1)] = (d, my_delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            assert row['r2'] == 0
            assert row['d'] <= cutoff
            assert row['d'] >= 0
            key = row['i'], row['r0'], row['r1']
            assert key in check
            assert check[key][0] <= cutoff
            assert check[key][0] >= 0
            assert abs(check[key][0] - row['d']) < 1e-8
            assert abs(check[key][1][0] - row['dx']) < 1e-8
            assert abs(check[key][1][1] - row['dy']) < 1e-8
            assert abs(check[key][1][2] - row['dz']) < 1e-8


def test_nlists_polyethylene4_9A():
    system = get_system_polyethylene4()
    nlists = NeighborLists(system)
    cutoff = 9*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in random.sample(xrange(system.natom), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(0, system.natom):
            delta = system.pos[i] - system.pos[j]
            for c in xrange(len(system.rvecs)):
                delta -= system.rvecs[c]*np.floor(np.dot(delta, system.gvecs[c]) + 0.5)
            for r0 in xrange(-3, 3):
                my_delta = delta + r0*system.rvecs[0]
                d = np.linalg.norm(my_delta)
                if d <= cutoff:
                    if (r0!=0) or (j>i):
                        check[(j, r0)] = (d, my_delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            assert row['r1'] == 0
            assert row['r2'] == 0
            assert row['d'] <= cutoff
            assert row['d'] >= 0
            key = row['i'], row['r0']
            assert key in check
            assert check[key][0] <= cutoff
            assert check[key][0] >= 0
            assert abs(check[key][0] - row['d']) < 1e-8
            assert abs(check[key][1][0] - row['dx']) < 1e-8
            assert abs(check[key][1][1] - row['dy']) < 1e-8
            assert abs(check[key][1][2] - row['dz']) < 1e-8


def test_nlists_quartz_9A():
    system = get_system_quartz()
    nlists = NeighborLists(system)
    cutoff = 9*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in random.sample(xrange(system.natom), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(0, system.natom):
            delta = system.pos[i] - system.pos[j]
            for c in xrange(len(system.rvecs)):
                delta -= system.rvecs[c]*np.floor(np.dot(delta, system.gvecs[c]) + 0.5)
            for r0 in xrange(-3, 3):
                for r1 in xrange(-3, 3):
                    for r2 in xrange(-3, 3):
                        my_delta = delta + r0*system.rvecs[0] + r1*system.rvecs[1] + r2*system.rvecs[2]
                        d = np.linalg.norm(my_delta)
                        if d <= cutoff:
                            if (r0!=0) or (r1!=0) or (r2!=0) or (j>i):
                                check[(j, r0, r1, r2)] = (d, my_delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            assert row['d'] <= cutoff
            assert row['d'] >= 0
            key = row['i'], row['r0'], row['r1'], row['r2']
            assert key in check
            assert check[key][0] <= cutoff
            assert check[key][0] >= 0
            assert abs(check[key][0] - row['d']) < 1e-8
            assert abs(check[key][1][0] - row['dx']) < 1e-8
            assert abs(check[key][1][1] - row['dy']) < 1e-8
            assert abs(check[key][1][2] - row['dz']) < 1e-8


def test_nlists_glycine_9A():
    system = get_system_glycine()
    nlists = NeighborLists(system)
    cutoff = 9*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in random.sample(xrange(system.natom), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(i+1, system.natom):
            delta = system.pos[i] - system.pos[j]
            d = np.linalg.norm(delta)
            if d <= cutoff:
                check[j] = (d, delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            assert row['r0'] == 0
            assert row['r1'] == 0
            assert row['r2'] == 0
            assert row['d'] <= cutoff
            assert row['d'] >= 0
            key = row['i']
            assert key in check
            assert check[key][0] <= cutoff
            assert check[key][0] >= 0
            assert abs(check[key][0] - row['d']) < 1e-8
            assert abs(check[key][1][0] - row['dx']) < 1e-8
            assert abs(check[key][1][1] - row['dy']) < 1e-8
            assert abs(check[key][1][2] - row['dz']) < 1e-8
