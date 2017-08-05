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
from __future__ import print_function

import numpy as np
from molmod import bond_length, bend_angle, bend_cos, dihed_angle, dihed_cos

from yaff import *

from yaff.test.common import get_system_quartz, get_system_peroxide, \
    get_system_mil53, get_system_water32, get_system_formaldehyde, \
    get_system_amoniak



def test_iclist_quartz_bonds():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    for i, j in system.bonds:
        iclist.add_ic(Bond(i, j))
    dlist.forward()
    iclist.forward()
    for row, (i, j) in enumerate(system.bonds):
        delta = system.pos[j] - system.pos[i]
        system.cell.mic(delta)
        assert abs(iclist.ictab[row]['value'] - bond_length([np.zeros(3, float), delta])[0]) < 1e-5


def test_iclist_quartz_bend_cos():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    angles = []
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    iclist.add_ic(BendCos(i0, i1, i2))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i1] - system.pos[i0]
        system.cell.mic(delta0)
        delta2 = system.pos[i1] - system.pos[i2]
        system.cell.mic(delta2)
        assert abs(iclist.ictab[row]['value'] - bend_cos([delta0, np.zeros(3, float), delta2])[0]) < 1e-5


def test_iclist_quartz_bend_angle():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    angles = []
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    iclist.add_ic(BendAngle(i0, i1, i2))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i1] - system.pos[i0]
        system.cell.mic(delta0)
        delta2 = system.pos[i1] - system.pos[i2]
        system.cell.mic(delta2)
        assert abs(iclist.ictab[row]['value'] - bend_angle([delta0, np.zeros(3, float), delta2])[0]) < 1e-5


def test_iclist_linear_bend_angle():
    numbers = np.array([1]*3)
    # Linear arrangement of 3 atoms, with a small distance between two atoms
    # This leads to round-off errors in calculation of the cosine of the angle
    pos = np.array([[0.0,0.0,0.0],[0.001,0.0,0.0],[1.0,0.0,0.0]])
    system = System(numbers, pos)
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    iclist.add_ic(BendAngle(0,1,2))
    dlist.forward()
    iclist.forward()
    # This test checks that our guard against round-off errors in the
    # calculation of the cosine of the angle works.
    theta = iclist.ictab[0]['value']
    assert np.isfinite(theta)


def test_iclist_peroxide_dihedral_cos():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        # The bonds are added randomly to get different situations in the delta list
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        iclist.add_ic(DihedCos(0,1,2,3))
        dlist.forward()
        iclist.forward()
        assert iclist.ictab[3]['kind']==3 #assert the third ic is DihedralCos
        assert abs(iclist.ictab[3]['value'] - dihed_cos(system.pos)[0]) < 1e-5


def test_iclist_peroxide_dihedral_angle():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        # The bonds are added randomly to get different situations in the delta list
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        iclist.add_ic(DihedAngle(0,1,2,3))
        dlist.forward()
        iclist.forward()
        assert iclist.ictab[3]['kind']==4 #assert the third ic is DihedralAngle
        assert abs(iclist.ictab[3]['value'] - dihed_angle(system.pos)[0]) < 1e-5



def test_iclist_grad_dihedral_cos_mil53():
    system = get_system_mil53()
    forbidden_dihedrals = [
        ["O_HY","AL","O_HY","AL"],
        ["O_HY","AL","O_HY","H_HY"],
        ["O_CA","AL","O_CA","C_CA"],
    ]
    idih = -1
    for i1, i2 in system.bonds:
        for i0 in system.neighs1[i1]:
            if i0==i2: continue
            for i3 in system.neighs1[i2]:
                if i3==i1: continue
                types = [system.get_ffatype(i0), system.get_ffatype(i1), system.get_ffatype(i2), system.get_ffatype(i3)]
                if types in forbidden_dihedrals or types[::-1] in forbidden_dihedrals: continue
                idih += 1
                dlist = DeltaList(system)
                iclist = InternalCoordinateList(dlist)
                iclist.add_ic(DihedCos(i0,i1,i2,i3))

                ic = iclist.ictab[0]
                dlist.forward()
                iclist.forward()
                ic['grad'] = 1.0
                iclist.back()
                cos = ic['value']
                grad_d0 = np.array([dlist.deltas[ic['i0']]['gx'], dlist.deltas[ic['i0']]['gy'], dlist.deltas[ic['i0']]['gz']])
                grad_d1 = np.array([dlist.deltas[ic['i1']]['gx'], dlist.deltas[ic['i1']]['gy'], dlist.deltas[ic['i1']]['gz']])
                grad_d2 = np.array([dlist.deltas[ic['i2']]['gx'], dlist.deltas[ic['i2']]['gy'], dlist.deltas[ic['i2']]['gz']])

                delta0 = system.pos[i0] - system.pos[i1]
                delta1 = system.pos[i2] - system.pos[i1]
                delta2 = system.pos[i3] - system.pos[i2]
                system.cell.mic(delta0)
                system.cell.mic(delta1)
                system.cell.mic(delta2)

                check_cos, check_grad = dihed_cos([delta0, np.zeros(3, float), delta1, delta1+delta2],deriv=1)
                check_grad_d0 = check_grad[0,:]
                check_grad_d1 = -check_grad[0,:] - check_grad[1,:]
                check_grad_d2 = check_grad[3,:]

                if not abs(ic['value'] - check_cos) < 1e-8:
                    raise AssertionError("Dihed cos (%s[%i],%s[%i],%s[%i],%s[%i]) should have value %10.9e, instead it is %10.9e" %(
                        system.get_ffatype(i0),i0,
                        system.get_ffatype(i1),i1,
                        system.get_ffatype(i2),i2,
                        system.get_ffatype(i3),i3,
                        check_cos,
                        cos
                    ))
                if not np.sqrt(sum( (grad_d0 - check_grad_d0)**2 )) < 1e-8:
                    raise AssertionError("Dihed cos (%s[%i],%s[%i],%s[%i],%s[%i]) should have delta0_grad [%12.9f,%12.9f,%12.9f], \n"
                                         "instead it is [%12.9f,%12.9f,%12.9f]" %(
                                            system.get_ffatype(i0),i0,
                                            system.get_ffatype(i1),i1,
                                            system.get_ffatype(i2),i2,
                                            system.get_ffatype(i3),i3,
                                            check_grad_d0[0], check_grad_d0[1], check_grad_d0[2],
                                            grad_d0[0], grad_d0[1], grad_d0[2],
                    ))
                if not np.sqrt(sum( (grad_d1 - check_grad_d1)**2 )) < 1e-8:
                    raise AssertionError("Dihed cos (%s[%i],%s[%i],%s[%i],%s[%i]) should have delta1_grad [%12.9f,%12.9f,%12.9f], \n"
                                         "instead it is [%12.9f,%12.9f,%12.9f]" %(
                                            system.get_ffatype(i0),i0,
                                            system.get_ffatype(i1),i1,
                                            system.get_ffatype(i2),i2,
                                            system.get_ffatype(i3),i3,
                                            check_grad_d1[0], check_grad_d1[1], check_grad_d1[2],
                                            grad_d1[0], grad_d1[1], grad_d1[2],
                    ))
                if not np.sqrt(sum( (grad_d2 - check_grad_d2)**2 )) < 1e-8:
                    raise AssertionError("Dihed cos (%s[%i],%s[%i],%s[%i],%s[%i]) should have delta2_grad [%12.9f,%12.9f,%12.9f], \n"
                                         "instead it is [%12.9f,%12.9f,%12.9f]" %(
                                            system.get_ffatype(i0),i0,
                                            system.get_ffatype(i1),i1,
                                            system.get_ffatype(i2),i2,
                                            system.get_ffatype(i3),i3,
                                            check_grad_d2[0], check_grad_d2[1], check_grad_d2[2],
                                            grad_d2[0], grad_d2[1], grad_d2[2],
                    ))


def test_iclist_ub_water():
    system = get_system_water32()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    ub = []
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    iclist.add_ic(UreyBradley(i0, i1, i2))
                    ub.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    for row, (i0, i1, i2) in enumerate(ub):
        delta = system.pos[i2] - system.pos[i0]
        system.cell.mic(delta)
        assert abs(iclist.ictab[row]['value'] - bond_length([np.zeros(3, float), delta])[0]) < 1e-5


def test_ic_list_dihedral_pernicious():
    system = get_system_peroxide()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    iclist.add_ic(DihedAngle(0,1,2,3))
    dlist.deltas[0] = (2.5645894177015855, -0.004063261303208772, 1.2798248429146284, 1, 2, 0.0, 0.0, 0.0)
    dlist.deltas[1] = (2.0394633015500796, -0.0032335148484117426, -1.98698089493469, 2, 3, 0.0, 0.0, 0.0)
    dlist.deltas[2] = (-1.8836397803889104, 0.0029844605526122576, -0.8613076025533011, 17, 3, 0.0, 0.0, 0.0)
    iclist.ictab[0] = (4, 0, -1, 1, 1, 2, -1, 0, 0, 0.0, 0.0)
    iclist.forward()
    assert abs(abs(iclist.ictab[0]['value']) - np.pi) < 1e-8


def test_oop_angle_formaldehyde():
    system = get_system_formaldehyde()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    iclist.add_ic(OopCos(2,3,1,0))
    iclist.add_ic(OopAngle(2,3,1,0))
    dlist.forward()
    iclist.forward()
    assert abs( iclist.ictab[0]['value'] - 1.0 ) < 1e-8
    assert abs( iclist.ictab[1]['value'] - 0.0 ) < 1e-8

def test_oop_meanangle_formaldehyde():
    system = get_system_formaldehyde()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    iclist.add_ic(OopMeanAngle(2,3,1,0))
    iclist.add_ic(OopAngle(1,2,3,0))
    iclist.add_ic(OopAngle(2,3,1,0))
    iclist.add_ic(OopAngle(3,1,2,0))
    dlist.forward()
    iclist.forward()
    mean = 0.0
    for i in range(4):
        assert abs( iclist.ictab[i]['value'] - 0.0) < 1e-8
        if i > 0: mean += iclist.ictab[i]['value']
    mean /= 3
    assert abs(iclist.ictab[0]['value'] - mean) < 1e-8

def test_oop_dist_formaldehyde():
    # All atoms in the y-z plane
    system = get_system_formaldehyde()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    iclist.add_ic(OopDist(2,3,1,0))
    dlist.forward()
    iclist.forward()
    assert abs( iclist.ictab[0]['value'] - 0.0 ) < 1e-8
    # Put the C atom out of the plane
    system = get_system_formaldehyde()
    system.pos[0,0] += 1.2*angstrom
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    # Add a dummy ic to the iclist, so some delta vectors are stored with a
    # minus sign. This enables to check that the (*ic).signs are implemented
    # correctly.
    iclist.add_ic(OopDist(1,3,2,0))
    iclist.add_ic(OopDist(2,3,1,0))
    dlist.forward()
    iclist.forward()
    assert abs( iclist.ictab[1]['value'] - 1.2*angstrom ) < 1e-8
    # Check if the distance is invariant for permutations of the first three atoms
    from itertools import permutations
    for perm in permutations((1,2,3)):
        iclist.add_ic(OopDist(perm[0],perm[1],perm[2],0))
    dlist.forward()
    iclist.forward()
    for ic in iclist.ictab:
        if ic['kind']==8:
            assert abs( abs(ic['value']) - 1.2*angstrom ) < 1e-8

def test_oop_meanangle_amoniak():
    system = get_system_amoniak()
    ics = [OopAngle(1,2,3,0), OopAngle(2,3,1,0), OopAngle(3,1,2,0)]
    angles = np.array([58.4158627745, 58.421383756, 58.4158570591])*deg
    #calculate mean of gradients of angles
    mean_value = 0.0
    mean_dgrad = {}
    for i, ic in enumerate(ics):
        print('Adding ic %i' %i)
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        iclist.add_ic(ic)
        dlist.forward()
        iclist.forward()
        assert abs(iclist.ictab[0]['value'] - angles[i]) < 1e-8
        mean_value += iclist.ictab[0]['value']/3.0
        iclist.ictab[0]['grad'] = 1.0/3.0
        iclist.back()
        for j in range(3):
            delta = dlist.deltas[j]
            key = '%i-%i' %(delta['i'], delta['j'])
            print('d(%s).g = ' %key, delta['gx'], delta['gy'], delta['gz'])
        print()
        for j in range(3):
            delta = dlist.deltas[j]
            key0 = '%i-%i' %(delta['i'], delta['j'])
            key1 = '%i-%i' %(delta['j'], delta['i'])
            if key0 in list(mean_dgrad.keys()):
                mean_dgrad[key0] += np.array([delta['gx'], delta['gy'], delta['gz']])
                print('mean_dgrad[%s]: ' %key0, mean_dgrad[key0])
            elif key1 in list(mean_dgrad.keys()):
                mean_dgrad[key1] -= np.array([delta['gx'], delta['gy'], delta['gz']])
                print('mean_dgrad[%s]: ' %key1, mean_dgrad[key1])
            else:
                mean_dgrad[key0] = np.array([delta['gx'], delta['gy'], delta['gz']])
                print('mean_dgrad[%s]: ' %key0, mean_dgrad[key0])
        print()
    print()
    #calculate gradient of meanangle
    print('Processing mean')
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    iclist.add_ic(OopMeanAngle(1,2,3,0))
    dlist.forward()
    iclist.forward()
    assert abs(iclist.ictab[0]['value'] - mean_value) < 1e-8
    iclist.ictab[0]['grad'] = 1.0
    iclist.back()
    for i in range(3):
        delta = dlist.deltas[i]
        key0 = '%i-%i' %(delta['i'], delta['j'])
        key1 = '%i-%i' %(delta['j'], delta['i'])
        print('d_gx, d_gy, d_gz (%s) = ' %(key0), delta['gx'], delta['gy'], delta['gz'])
        if key0 in mean_dgrad.keys():
            mean = mean_dgrad[key0]
        elif key1 in mean_dgrad.keys():
            mean = -mean_dgrad[key1]
        else:
            raise AssertionError('Delta %s not found in dlist of system with OopMeanAngle' %key0)
        assert abs(delta['gx'] - mean[0]) < 1e-8
        assert abs(delta['gy'] - mean[1]) < 1e-8
        assert abs(delta['gz'] - mean[2]) < 1e-8
