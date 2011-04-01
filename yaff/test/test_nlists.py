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
from common import get_system_h2o32

from yaff import *


def test_nlists_h2o32_4A():
    system = get_system_h2o32()
    nlists = NeighborLists(system)
    cutoff = 4*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in random.sample(xrange(system.size), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(system.size):
            delta = system.pos[i] - system.pos[j]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            d = np.linalg.norm(delta)
            if d <= cutoff:
                check[j] = (d, delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            assert row[0] in check
            assert abs(check[row[0]][0]) <= cutoff
            assert abs(check[row[0]][0] - row[1]) < 1e-8
            assert abs(check[row[0]][1] - row[2]).max() < 1e-8
            assert (row[3] == 0).all()


def test_nlists_h2o32_9A():
    system = get_system_h2o32()
    nlists = NeighborLists(system)
    cutoff = 9*angstrom
    nlists.request_cutoff(cutoff)
    nlists.update()
    for i in random.sample(xrange(system.size), 5):
        # compute the distances in the neighborlist manually and check.
        check = {}
        for j in xrange(system.size):
            delta = system.pos[i] - system.pos[j]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            assert abs(delta).max() < 0.5*9.865*angstrom
            for l0 in xrange(-1, 2):
                for l1 in xrange(-1, 2):
                    for l2 in xrange(-1, 2):
                        my_delta = delta + np.array([l0,l1,l2])*9.865*angstrom
                        d = np.linalg.norm(my_delta)
                        if d <= cutoff:
                            check[(j, l0, l1, l2)] = (d, my_delta)
        # compare
        assert len(nlists[i]) == len(check)
        for row in nlists[i]:
            key = row[0], row[3][0], row[3][1], row[3][2]
            assert key in check
            assert abs(check[key][0]) <= cutoff
            assert abs(check[key][0] - row[1]) < 1e-8
            assert abs(check[key][1] - row[2]).max() < 1e-8
