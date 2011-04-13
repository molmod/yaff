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
from molmod import bond_length, bend_angle, bend_cos

from yaff import *

from common import get_system_quartz


def test_vlist_quartz_bonds():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    for i, j in system.topology.bonds:
        vlist.add_term(Harmonic(2.3, 3.04+0.1*i, Bond(i, j)))
    assert dlist.ndelta == len(system.topology.bonds)
    assert iclist.nic == len(system.topology.bonds)
    assert vlist.nv == len(system.topology.bonds)
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    for i, j in system.topology.bonds:
        delta = system.pos[i] - system.pos[j]
        for c in xrange(len(system.rvecs)):
            delta -= system.rvecs[c]*np.ceil(np.dot(delta, system.gvecs[c]) - 0.5)
        d = np.linalg.norm(delta)
        check_energy += 0.5*2.3*(d - 3.04-0.1*i)**2
    assert abs(energy - check_energy) < 1e-8


def test_vlist_quartz_bend_cos():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    angles = []
    for i1 in xrange(system.natom):
        for i0 in system.topology.neighs1[i1]:
            for i2 in system.topology.neighs1[i1]:
                if i0 > i2:
                    vlist.add_term(Harmonic(1.1+0.01*i0, -0.2, BendCos(i0, i1, i2)))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i0] - system.pos[i1]
        for c in xrange(len(system.rvecs)):
            delta0 -= system.rvecs[c]*np.ceil(np.dot(delta0, system.gvecs[c]) - 0.5)
        delta2 = system.pos[i2] - system.pos[i1]
        for c in xrange(len(system.rvecs)):
            delta2 -= system.rvecs[c]*np.ceil(np.dot(delta2, system.gvecs[c]) - 0.5)
        c = bend_cos(delta0, np.zeros(3, float), delta2)[0]
        check_energy += 0.5*(1.1+0.01*i0)*(c+0.2)**2
    assert abs(energy - check_energy) < 1e-8


def test_vlist_quartz_bend_angle():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    angles = []
    for i1 in xrange(system.natom):
        for i0 in system.topology.neighs1[i1]:
            for i2 in system.topology.neighs1[i1]:
                if i0 > i2:
                    vlist.add_term(Harmonic(1.5, 2.0+0.01*i2, BendAngle(i0, i1, i2)))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i0] - system.pos[i1]
        for c in xrange(len(system.rvecs)):
            delta0 -= system.rvecs[c]*np.ceil(np.dot(delta0, system.gvecs[c]) - 0.5)
        delta2 = system.pos[i2] - system.pos[i1]
        for c in xrange(len(system.rvecs)):
            delta2 -= system.rvecs[c]*np.ceil(np.dot(delta2, system.gvecs[c]) - 0.5)
        angle = bend_angle(delta0, np.zeros(3, float), delta2)[0]
        check_energy += 0.5*1.5*(angle-(2.0+0.01*i2))**2
    assert abs(energy - check_energy) < 1e-8
