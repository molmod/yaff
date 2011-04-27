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


import numpy as np
from molmod import bond_length, bend_angle, bend_cos, dihed_angle, dihed_cos
from nose.plugins.skip import SkipTest

from yaff import *

from common import get_system_quartz, get_system_peroxide


def test_iclist_quartz_bonds():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    for i, j in system.topology.bonds:
        iclist.add_ic(Bond(i, j))
    dlist.forward()
    iclist.forward()
    for row, (i, j) in enumerate(system.topology.bonds):
        delta = system.pos[i] - system.pos[j]
        for c in xrange(system.cell.nvec):
            delta -= system.cell.rvecs[c]*np.ceil(np.dot(delta, system.cell.gvecs[c]) - 0.5)
        assert abs(iclist.ictab[row]['value'] - bond_length(np.zeros(3, float), delta)[0]) < 1e-5


def test_iclist_quartz_bend_cos():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    angles = []
    for i1 in xrange(system.natom):
        for i0 in system.topology.neighs1[i1]:
            for i2 in system.topology.neighs1[i1]:
                if i0 > i2:
                    iclist.add_ic(BendCos(i0, i1, i2))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i0] - system.pos[i1]
        for c in xrange(system.cell.nvec):
            delta0 -= system.cell.rvecs[c]*np.ceil(np.dot(delta0, system.cell.gvecs[c]) - 0.5)
        delta2 = system.pos[i2] - system.pos[i1]
        for c in xrange(system.cell.nvec):
            delta2 -= system.cell.rvecs[c]*np.ceil(np.dot(delta2, system.cell.gvecs[c]) - 0.5)
        assert abs(iclist.ictab[row]['value'] - bend_cos(delta0, np.zeros(3, float), delta2)[0]) < 1e-5


def test_iclist_quartz_bend_angle():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    angles = []
    for i1 in xrange(system.natom):
        for i0 in system.topology.neighs1[i1]:
            for i2 in system.topology.neighs1[i1]:
                if i0 > i2:
                    iclist.add_ic(BendAngle(i0, i1, i2))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i0] - system.pos[i1]
        for c in xrange(system.cell.nvec):
            delta0 -= system.cell.rvecs[c]*np.ceil(np.dot(delta0, system.cell.gvecs[c]) - 0.5)
        delta2 = system.pos[i2] - system.pos[i1]
        for c in xrange(system.cell.nvec):
            delta2 -= system.cell.rvecs[c]*np.ceil(np.dot(delta2, system.cell.gvecs[c]) - 0.5)
        assert abs(iclist.ictab[row]['value'] - bend_angle(delta0, np.zeros(3, float), delta2)[0]) < 1e-5

def test_iclist_peroxide_dihedral_cos():
    number_of_tests=50
    for i in xrange(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.topology.bonds or (i1,i0) in system.topology.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        iclist.add_ic(DihedCos(0,1,2,3))
        dlist.forward()
        iclist.forward()
        assert iclist.ictab[3]['kind']==3 #assert the third ic is DihedralCos
        assert abs(iclist.ictab[3]['value'] - dihed_cos(system.pos[0],system.pos[1],system.pos[2],system.pos[3])[0]) < 1e-5

def test_iclist_peroxide_dihedral_angle():
    number_of_tests=50
    for i in xrange(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.topology.bonds or (i1,i0) in system.topology.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        iclist.add_ic(DihedAngle(0,1,2,3))
        dlist.forward()
        iclist.forward()
        assert iclist.ictab[3]['kind']==4 #assert the third ic is DihedralAngle
        assert abs(iclist.ictab[3]['value'] - dihed_angle(system.pos[0],system.pos[1],system.pos[2],system.pos[3])[0]) < 1e-5
