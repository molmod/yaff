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
from molmod import bend_angle, bend_cos, dihed_angle, dihed_cos
from nose.plugins.skip import SkipTest

from yaff import *

from yaff.test.common import get_system_quartz, get_system_water32, \
    get_system_2T, get_system_peroxide, get_system_mil53, get_system_formaldehyde
from yaff.pes.test.common import check_gpos_part, check_vtens_part


def test_vlist_quartz_bonds():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    for i, j in system.bonds:
        vlist.add_term(Harmonic(2.3, 3.04+0.1*i, Bond(i, j)))
    assert dlist.ndelta == len(system.bonds)
    assert iclist.nic == len(system.bonds)
    assert vlist.nv == len(system.bonds)
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    counter = 0
    for i, j in system.bonds:
        delta = system.pos[i] - system.pos[j]
        system.cell.mic(delta)
        d = np.linalg.norm(delta)
        check_term = 0.5*2.3*(d - 3.04-0.1*i)**2
        assert abs(check_term - vlist.vtab[counter]['energy']) < 1e-10
        check_energy += check_term
        counter += 1
    assert abs(energy - check_energy) < 1e-8


def test_vlist_quartz_bonds_fues():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    for i, j in system.bonds:
        vlist.add_term(Fues(2.3, 3.04+0.1*i, Bond(i, j)))
    assert dlist.ndelta == len(system.bonds)
    assert iclist.nic == len(system.bonds)
    assert vlist.nv == len(system.bonds)
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    counter = 0
    for i, j in system.bonds:
        delta = system.pos[i] - system.pos[j]
        system.cell.mic(delta)
        d = np.linalg.norm(delta)
        rv = 3.04+0.1*i
        check_term = 0.5*2.3*rv**2*(1.0 + rv/d*(rv/d - 2.0))
        if not abs(check_term - vlist.vtab[counter]['energy']) < 1e-10:
            raise AssertionError("Error in energy of bond(%i,%i): energy = %.15f instead energy = %.15f" %(i,j,check_term,vlist.vtab[counter]['energy']))
        check_energy += check_term
        counter += 1
    assert abs(energy - check_energy) < 1e-8


def test_vlist_quartz_bend_cos():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    angles = []
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    vlist.add_term(Harmonic(1.1+0.01*i0, -0.2, BendCos(i0, i1, i2)))
                    angles.append((i0, i1, i2))
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    counter = 0
    for row, (i0, i1, i2) in enumerate(angles):
        delta0 = system.pos[i0] - system.pos[i1]
        system.cell.mic(delta0)
        delta2 = system.pos[i2] - system.pos[i1]
        system.cell.mic(delta2)
        c = bend_cos([delta0, np.zeros(3, float), delta2])[0]
        check_term = 0.5*(1.1+0.01*i0)*(c+0.2)**2
        assert abs(check_term - vlist.vtab[counter]['energy']) < 1e-10
        check_energy += check_term
        counter += 1
    assert abs(energy - check_energy) < 1e-8


def test_vlist_quartz_bend_angle():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    angles = []
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
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
        system.cell.mic(delta0)
        delta2 = system.pos[i2] - system.pos[i1]
        system.cell.mic(delta2)
        angle = bend_angle([delta0, np.zeros(3, float), delta2])[0]
        check_energy += 0.5*1.5*(angle-(2.0+0.01*i2))**2
    assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_cos():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        vlist.add_term(Harmonic(1.1, -0.2 , DihedCos(0,1,2,3)))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        cos = dihed_cos(system.pos)[0]
        check_energy = 0.5*1.1*(cos+0.2)**2
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_cos_chebychev1():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        amp = np.random.normal(0, 1)
        sign = np.random.randint(2)
        if sign==0: sign = -1
        vlist.add_term(Chebychev1(amp, DihedCos(0,1,2,3), sign=sign))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        psi = dihed_angle(system.pos)[0]
        check_energy = 0.5*amp*(1.0+sign*np.cos(psi))
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_cos_chebychev2():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        amp = np.random.normal(0, 1)
        sign = np.random.randint(2)
        if sign==0: sign = -1
        vlist.add_term(Chebychev2(amp, DihedCos(0,1,2,3), sign=sign))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        psi = dihed_angle(system.pos)[0]
        check_energy = 0.5*amp*(1.0+sign*np.cos(2*psi))
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_cos_chebychev3():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        amp = np.random.normal(0, 1)
        sign = np.random.randint(2)
        if sign==0: sign = -1
        vlist.add_term(Chebychev3(amp, DihedCos(0,1,2,3), sign=sign))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        psi = dihed_angle(system.pos)[0]
        check_energy = 0.5*amp*(1.0+sign*np.cos(3*psi))
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_cos_chebychev4():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        amp = np.random.normal(0, 1)
        sign = np.random.randint(2)
        if sign==0: sign = -1
        vlist.add_term(Chebychev4(amp, DihedCos(0,1,2,3), sign=sign))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        psi = dihed_angle(system.pos)[0]
        check_energy = 0.5*amp*(1.0+sign*np.cos(4*psi))
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_cos_chebychev6():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        amp = np.random.normal(0, 1)
        sign = np.random.randint(2)
        if sign==0: sign = -1
        vlist.add_term(Chebychev6(amp, DihedCos(0,1,2,3), sign=sign))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        psi = dihed_angle(system.pos)[0]
        check_energy = 0.5*amp*(1.0+sign*np.cos(6*psi))
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_angle():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        vlist.add_term(Harmonic(1.5, 0.1 , DihedAngle(0,1,2,3)))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        angle = dihed_angle(system.pos)[0]
        check_energy = 0.5*1.5*(angle-0.1)**2
        assert abs(energy - check_energy) < 1e-8


def test_vlist_peroxide_dihed_angle_cosine():
    number_of_tests=50
    for i in range(number_of_tests):
        system = get_system_peroxide()
        dlist = DeltaList(system)
        iclist = InternalCoordinateList(dlist)
        vlist = ValenceList(iclist)
        bonds=[]
        while len(bonds)<3:
            i0, i1 = [int(x) for x in np.random.uniform(low=0,high=4,size=2)] #pick 2 random atoms
            if i0==i1 or (i0,i1) in bonds or (i1,i0) in bonds: continue
            if (i0,i1) in system.bonds or (i1,i0) in system.bonds:
                iclist.add_ic(Bond(i0,i1))
                bonds.append((i0,i1))
        mult = np.random.randint(1, 4)
        amp = np.random.normal(0, 1)
        phi0 = np.random.uniform(0, 2*np.pi)
        vlist.add_term(Cosine(mult, amp, phi0, DihedAngle(0,1,2,3)))
        dlist.forward()
        iclist.forward()
        energy = vlist.forward()
        # calculate energy manually
        angle = dihed_angle(system.pos)[0]
        check_energy = 0.5*amp*(1-np.cos(mult*(angle-phi0)))
        assert abs(energy - check_energy) < 1e-8


def test_vlist_polyfour_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(PolyFour([1.1+0.01*i, 0.8+0.01*j, 0.6+0.01*i, 0.4+0.01*j], Bond(i, j)))
    energy = part.compute()
    check_energy = 0.0
    for i, j in system.bonds:
        delta = system.pos[j] - system.pos[i]
        system.cell.mic(delta)
        bond = np.linalg.norm(delta)
        check_energy += (1.1+0.01*i)*bond + (0.8+0.01*j)*bond**2 + (0.6+0.01*i)*bond**3 + (0.4+0.01*j)*bond**4
    assert abs(energy - check_energy) < 1e-8


def test_vlist_cross_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for j in range(system.natom):
        if len(system.neighs1[j])==2:
            i, k = system.neighs1[j]
            part.add_term(Cross(
                    1.2,
                    1.7 + 0.01*i,
                    1.9 + 0.01*k,
                    Bond(i, j),
                    Bond(j, k),
            ))
    energy = part.compute()
    check_energy = 0.0
    for j in range(system.natom):
        if len(system.neighs1[j])==2:
            i, k = system.neighs1[j]
            delta0 = system.pos[j] - system.pos[i]
            delta1 = system.pos[k] - system.pos[j]
            system.cell.mic(delta0)
            system.cell.mic(delta1)
            bond0 = np.linalg.norm(delta0)
            bond1 = np.linalg.norm(delta1)
            check_energy += 1.2*(bond0 - 1.7 - 0.01*i)*(bond1 - 1.9 - 0.01*k)
    assert abs(energy - check_energy) < 1e-8


def test_vlist_dihedral_cos_mil53():
    system = get_system_mil53()
    part = ForcePartValence(system)
    for i1, i2 in system.bonds:
        for i0 in system.neighs1[i1]:
            if i0==i2: continue
            for i3 in system.neighs1[i2]:
                if i3==i1: continue
                fc = 2.1 + 0.01*(0.3*i1 + 0.7*i2)
                part.add_term(PolyFour([0.0,-2.0*fc,0.0,0.0],DihedCos(i0,i1,i2,i3)))
    energy = part.compute()
    check_energy = 0.0
    for i1, i2 in system.bonds:
        for i0 in system.neighs1[i1]:
            if i0==i2: continue
            for i3 in system.neighs1[i2]:
                if i3==i1: continue
                fc = 2.1 + 0.01*(0.3*i1 + 0.7*i2)
                delta0 = system.pos[i0] - system.pos[i1]
                delta1 = system.pos[i2] - system.pos[i1]
                delta2 = system.pos[i3] - system.pos[i2]
                system.cell.mic(delta0)
                system.cell.mic(delta1)
                system.cell.mic(delta2)
                cos = dihed_cos([delta0, np.zeros(3, float), delta1, delta1+delta2])[0]
                check_energy += -2.0*fc*cos**2
    if not abs(energy - check_energy) < 1e-8:
        raise AssertionError("Energy should be %10.9e, instead it is %10.9e" %(check_energy, energy))


def test_vlist_quartz_morse():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    for i, j in system.bonds:
        vlist.add_term(Morse(2.3+i, 0.5, 2.0, Bond(i, j)))
    assert dlist.ndelta == len(system.bonds)
    assert iclist.nic == len(system.bonds)
    assert vlist.nv == len(system.bonds)
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    # compute energy manually
    check_energy = 0.0
    counter = 0
    for i, j in system.bonds:
        delta = system.pos[i] - system.pos[j]
        system.cell.mic(delta)
        d = np.linalg.norm(delta)
        check_term = (2.3+i)*(np.exp(-2*0.5*(d-2.0)) - 2.0*np.exp(-0.5*(d-2.0)))
        assert abs(check_term - vlist.vtab[counter]['energy']) < 1e-10
        check_energy += check_term
        counter += 1
    assert abs(energy - check_energy) < 1e-8


def test_gpos_vtens_bond_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(Harmonic(0.3, 1.7, Bond(i, j)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_bond_fues_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(Fues(0.3, 1.7, Bond(i, j)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_bend_cos_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(Harmonic(1.1+0.01*i0, -0.2, BendCos(i0, i1, i2)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_bend_angle_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(Harmonic(1.5, 2.0+0.01*i2, BendAngle(i0, i1, i2)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_cos_peroxide():
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Harmonic(1.1, -0.2, DihedCos(0,1,2,3)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_cos_chebychev1_peroxide():
    #Test for positive sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev1(1.5, DihedCos(0,1,2,3), sign=1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    #Test for negative sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev1(1.5, DihedCos(0,1,2,3), sign=-1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_cos_chebychev2_peroxide():
    #Test for positive sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev2(1.5, DihedCos(0,1,2,3), sign=1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    #Test for negative sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev2(1.5, DihedCos(0,1,2,3), sign=-1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_cos_chebychev3_peroxide():
    #Test for positive sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev3(1.5, DihedCos(0,1,2,3), sign=1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    #Test for negative sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev3(1.5, DihedCos(0,1,2,3), sign=-1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_cos_chebychev4_peroxide():
    #Test for positive sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev4(1.5, DihedCos(0,1,2,3), sign=1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    #Test for negative sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev4(1.5, DihedCos(0,1,2,3), sign=-1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_cos_chebychev6_peroxide():
    #Test for positive sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev6(1.5, DihedCos(0,1,2,3), sign=1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    #Test for negative sign in polynomial
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Chebychev6(1.5, DihedCos(0,1,2,3), sign=-1))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_angle_peroxide():
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Harmonic(1.5, 1.0, DihedAngle(0,1,2,3)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihed_angle_cosine_peroxide():
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Cosine(3, 1.5, 2*np.pi/3, DihedAngle(0,1,2,3)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_2T():
    system = get_system_2T()
    rv_table = {
        ('H', 'Si', 'H'):  1.80861,
        ('Si', 'O'):       3.24970,
        ('H', 'O'):        1.96022,
        ('O', 'Si', 'O'):  2.06457,
        ('O', 'Si'):       3.24970,
        ('O', 'Si', 'H'):  1.85401,
        ('Si', 'O', 'Si'): 1.80173,
        ('Si', 'O', 'H'):  1.55702,
        ('H', 'O', 'Si'):  1.55702,
        ('Si', 'H'):       2.87853,
        ('H', 'Si'):       2.87853,
        ('H', 'Si', 'O'):  1.85401,
        ('O', 'H'):        1.96022,
    }
    fc_table = {
        ('H', 'Si', 'H'):  0.09376,
        ('Si', 'O'):       0.30978,
        ('H', 'O'):        0.60322,
        ('O', 'Si', 'O'):  0.19282,
        ('O', 'Si'):       0.30978,
        ('O', 'Si', 'H'):  0.11852,
        ('Si', 'O', 'Si'): 0.00751,
        ('Si', 'O', 'H'):  0.04271,
        ('H', 'O', 'Si'):  0.04271,
        ('Si', 'H'):       0.18044,
        ('H', 'Si'):       0.18044,
        ('H', 'Si', 'O'):  0.11852,
        ('O', 'H'):        0.60322,
    }

    part = ForcePartValence(system)
    for i, j in system.bonds:
        key = system.get_ffatype(i), system.get_ffatype(j)
        part.add_term(Harmonic(fc_table[key], rv_table[key], Bond(i, j)))
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                key = system.get_ffatype(i0), system.get_ffatype(i1), system.get_ffatype(i2)
                if i0 > i2:
                    part.add_term(Harmonic(fc_table[key], rv_table[key], BendAngle(i0, i1, i2)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
    # same test but with Fues bonds instead of Harmonic bonds
    part = ForcePartValence(system)
    for i, j in system.bonds:
        key = system.get_ffatype(i), system.get_ffatype(j)
        part.add_term(Fues(fc_table[key], rv_table[key], Bond(i, j)))
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                key = system.get_ffatype(i0), system.get_ffatype(i1), system.get_ffatype(i2)
                if i0 > i2:
                    part.add_term(Harmonic(fc_table[key], rv_table[key], BendAngle(i0, i1, i2)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_quartz():
    system = get_system_quartz()
    rv_table = {
        ('Si', 'O'):       3.24970,
        ('O', 'Si', 'O'):  2.06457,
        ('Si', 'O', 'Si'): 1.80173,
    }
    fc_table = {
        ('Si', 'O'):       0.30978,
        ('O', 'Si', 'O'):  0.19282,
        ('Si', 'O', 'Si'): 0.00751,
    }

    part = ForcePartValence(system)
    for i, j in system.bonds:
        key = system.get_ffatype(i), system.get_ffatype(j)
        part.add_term(Harmonic(fc_table[key], rv_table[key], Bond(i, j)))
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                key = system.get_ffatype(i0), system.get_ffatype(i1), system.get_ffatype(i2)
                if i0 > i2:
                    part.add_term(Harmonic(fc_table[key], rv_table[key], BendAngle(i0, i1, i2)))

    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_polyfour_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(PolyFour([-0.5, 0.3, -0.16, 0.09], Bond(i, j)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_cross_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for j in range(system.natom):
        if len(system.neighs1[j])==2:
            i, k = system.neighs1[j]
            part.add_term(Cross(
                    1.2,
                    1.7,
                    1.9,
                    Bond(i, j),
                    Bond(j, k),
            ))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_dihedral_cos_mil53():
    system = get_system_mil53()
    forbidden_dihedrals = [
        ["O_HY","AL","O_HY","AL"],
        ["O_HY","AL","O_HY","H_HY"],
        ["O_CA","AL","O_CA","C_CA"],
        ["O_CA","AL","O_HY","H_HY"],
        ["H_PH","C_PH","C_PC","C_PH"],
        ["H_PH","C_PH","C_PC","C_CA"],
        ["C_PH","C_PH","C_PC","C_PH"],
        ["C_PH","C_PH","C_PC","C_CA"],
        ["C_PC","C_PH","C_PH","H_PH"],
        ["C_PC","C_PH","C_PH","C_PC"],
        ["H_PH","C_PH","C_PH","H_PH"],
        ["C_PH","C_PC","C_CA","O_CA"],
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
                fc = 2.1 + 0.01*(0.3*i1 + 0.7*i2)
                part = ForcePartValence(system)
                part.add_term(PolyFour([-2.0*fc,0.0001,0.0,0.0],DihedCos(i0,i1,i2,i3)))
                check_gpos_part(system, part)
                check_vtens_part(system, part)


def test_gpos_vtens_ub_water():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(Harmonic(2.1,2.0*angstrom,UreyBradley(i0, i1, i2)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_morse_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(Morse(0.3, 1.7, 2.0, Bond(i, j)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_mm3quartic_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(MM3Quartic(1.5, 2.0+0.01*i, Bond(i, j)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_mm3benda_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in range(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(MM3Bend(1.5, 2.0+0.01*i2, BendAngle(i0, i1, i2)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_zero_dihed_steven():
    pos = np.array([
       [  3.99364178360816302e+00,   6.30763314754801546e-02,  -3.46387534695341159e+00],
       [  8.25853872852823123e+00,  -2.56426319334358244e+00,   2.79913814647939019e-01],
       [  6.27951244286421861e+00,  -1.73747102970232015e+00,  -2.05686744048685455e+00],
       [  8.12025850622788603e+00,  -6.72445448176343996e-01,   2.91920676811204993e+00],
    ])
    numbers = np.array([1, 2, 3, 4])
    system = System(numbers, pos)
    fp = ForcePartValence(system)
    fp.add_term(Cosine(3, 0.001, 0.0, DihedAngle(0, 1, 2, 3)))
    gpos = np.zeros(pos.shape)
    fp.compute(gpos)
    assert fp.iclist.ictab[0]['value'] == 0.0
    assert fp.iclist.ictab[0]['grad'] == 0.0
    assert not np.isnan(gpos).any()


def test_pi_dihed_steven():
    raise SkipTest('Current implementation of dihedral angle is numerically instable for derivatives close to 0 and 180 deg.')
    pos0 = np.array([
        [-1.569651557428415,  2.607830491228437,  0.147778432480783],
        [ 0.232512681083857,  0.525428350542485, -0.040349603247728],
        [ 2.731099007352725,  1.143841961682292, -0.263793263490733],
        [ 4.427734700284959, -0.813154425943770, -0.440868229620179],
    ])
    pos1 = np.array([
        [-1.569651087819148,  2.607822317490105,  0.147780336986862],
        [ 0.232512798189943,  0.525437313611198, -0.040362705978732],
        [ 2.731087893401052,  1.143824270036481, -0.263785656061896],
        [ 4.427712083945147, -0.813160470365458, -0.440873675490262],

    ])

    def helper(pos):
        numbers = np.array([1, 2, 3, 4])
        system = System(numbers, pos)
        fp = ForcePartValence(system)
        fp.add_term(Cosine(3, 0.1, 1.0, DihedAngle(0, 1, 2, 3)))
        fp.dlist.forward()
        fp.iclist.forward()
        fp.iclist.ictab['grad'][0] = 1.0
        fp.iclist.back()
        gpos = np.zeros(pos.shape)
        fp.dlist.back(gpos, None)
        print(fp.iclist.ictab[0])
        return gpos

    gpos0 = helper(pos0)
    gpos1 = helper(pos1)

    print(gpos0)
    print(gpos1)
    print(gpos1/gpos0)
    assert abs(gpos0 - gpos1).max() < 1e-3


def test_inversion_formaldehyde():
    # Test for an inversion term made by combining a Chebychev1 energy term with
    # an out-of-plane cosine.
    oop_fc = 1.0
    system = get_system_formaldehyde()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    vlist = ValenceList(iclist)
    #Add a term for all out-of-plane angles with the carbon as center
    vlist.add_term(Chebychev1(oop_fc, OopCos(2,3,1,0)))
    vlist.add_term(Chebychev1(oop_fc, OopCos(1,2,3,0)))
    vlist.add_term(Chebychev1(oop_fc, OopCos(1,3,2,0)))
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    #For a planar molecule the energy should be zero
    check_energy = 0.0
    assert abs(energy - check_energy) < 1e-8
    #Now put the carbon out of the plane; the energy shoud rise
    system.pos[0,0] += 1.00000000*angstrom
    dlist.forward()
    iclist.forward()
    energy = vlist.forward()
    #Calculate the energy manually
    check_energy = 0.0
    for term in iclist.ictab:
        #Select the terms corresponding to oop cosines
        if term[0]==6:
            check_energy += 0.5*oop_fc*(1.0-term['value'])
    assert abs(energy - check_energy) < 1e-8


def get_ff_formaldehyde():
    # This forcefield is based on UFF, but we only really care about
    # out-of-plane angles
    system = get_system_formaldehyde()
    # Move the C atom to make the molecule non-planar
    system.pos[0,0] += 0.2*angstrom
    # Valence part
    part_valence = ForcePartValence(system)
    # Harmonic bonds
    part_valence.add_term(Harmonic(3.371456e3*kjmol/angstrom**2, 1.219*angstrom, Bond(0, 1)))
    part_valence.add_term(Harmonic(1.484153e3*kjmol/angstrom**2, 1.084*angstrom, Bond(0, 2)))
    part_valence.add_term(Harmonic(1.484153e3*kjmol/angstrom**2, 1.084*angstrom, Bond(0, 3)))
    # Harmonic bends
    part_valence.add_term(Harmonic(6.91928e2*kjmol, np.cos(120.0*deg), BendCos(1,0,2)))
    part_valence.add_term(Harmonic(6.91928e2*kjmol, np.cos(120.0*deg), BendCos(1,0,3)))
    part_valence.add_term(Harmonic(2.57789e2*kjmol, np.cos(120.0*deg), BendCos(2,0,3)))
    # Out-of-plane angles
    part_valence.add_term(Chebychev1(6.978*kjmol, OopCos(2,3,1,0)))
    part_valence.add_term(Chebychev1(6.978*kjmol, OopCos(1,2,3,0)))
    part_valence.add_term(Chebychev1(6.978*kjmol, OopCos(1,3,2,0)))
    return ForceField(system, [part_valence], None)


def test_opt_formaldehyde():
    ff = get_ff_formaldehyde()
    opt = QNOptimizer(CartesianDOF(ff))
    opt.run(100)
    #Check if all out-of-plane angles go to zero (so their cosine goes to 1)
    for ic in ff.part_valence.iclist.ictab:
        if ic['kind'] == 6:
            assert abs(ic['value'] - 1.0) < 1e-6


def test_gpos_vtens_oopcos_formaldehyde():
    system = get_system_formaldehyde()
    part = ForcePartValence(system)
    part.add_term(Harmonic(2.1,0.0*angstrom,OopCos(2,3,1,0)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)


def test_gpos_vtens_oopdist_formaldehyde():
    system = get_system_formaldehyde()
    system.pos[0,0] += 0.0*angstrom
    part = ForcePartValence(system)
    part.add_term(Harmonic(0.0,0.0*angstrom,OopDist(2,3,1,0)))
    check_gpos_part(system, part)
    check_vtens_part(system, part)
