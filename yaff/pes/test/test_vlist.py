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
from molmod import bend_angle, bend_cos, dihed_angle, dihed_cos

from yaff import *

from yaff.test.common import get_system_quartz, get_system_water32, \
    get_system_2T, get_system_peroxide, get_system_mil53
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
    for i1 in xrange(system.natom):
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
    for i1 in xrange(system.natom):
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
    for i in xrange(number_of_tests):
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


def test_vlist_peroxide_dihed_angle():
    number_of_tests=50
    for i in xrange(number_of_tests):
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
    for j in xrange(system.natom):
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
    for j in xrange(system.natom):
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


def test_gpos_vtens_bond_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(Harmonic(0.3, 1.7, Bond(i, j)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-7)


def test_gpos_vtens_bond_fues_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(Fues(0.3, 1.7, Bond(i, j)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-7)


def test_gpos_vtens_bend_cos_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in xrange(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(Harmonic(1.1+0.01*i0, -0.2, BendCos(i0, i1, i2)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-8)


def test_gpos_vtens_bend_angle_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in xrange(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(Harmonic(1.5, 2.0+0.01*i2, BendAngle(i0, i1, i2)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-8)


def test_gpos_vtens_dihed_cos_peroxide():
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Harmonic(1.1, -0.2, DihedCos(0,1,2,3)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-8)


def test_gpos_vtens_dihed_angle_peroxide():
    system = get_system_peroxide()
    part = ForcePartValence(system)
    part.add_term(Harmonic(1.5, 1.0, DihedAngle(0,1,2,3)))
    check_gpos_part(system, part, 1e-9)
    check_vtens_part(system, part, 1e-8)


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
        key = system.ffatypes[i], system.ffatypes[j]
        part.add_term(Harmonic(fc_table[key], rv_table[key], Bond(i, j)))
    for i1 in xrange(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                key = system.ffatypes[i0], system.ffatypes[i1], system.ffatypes[i2]
                if i0 > i2:
                    part.add_term(Harmonic(fc_table[key], rv_table[key], BendAngle(i0, i1, i2)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-10)
    # same test but with Fues bonds instead of Harmonic bonds
    part = ForcePartValence(system)
    for i, j in system.bonds:
        key = system.ffatypes[i], system.ffatypes[j]
        part.add_term(Fues(fc_table[key], rv_table[key], Bond(i, j)))
    for i1 in xrange(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                key = system.ffatypes[i0], system.ffatypes[i1], system.ffatypes[i2]
                if i0 > i2:
                    part.add_term(Harmonic(fc_table[key], rv_table[key], BendAngle(i0, i1, i2)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-10)


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
        key = system.ffatypes[i], system.ffatypes[j]
        part.add_term(Harmonic(fc_table[key], rv_table[key], Bond(i, j)))
    for i1 in xrange(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                key = system.ffatypes[i0], system.ffatypes[i1], system.ffatypes[i2]
                if i0 > i2:
                    part.add_term(Harmonic(fc_table[key], rv_table[key], BendAngle(i0, i1, i2)))

    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-10)


def test_gpos_vtens_polyfour_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i, j in system.bonds:
        part.add_term(PolyFour([-0.5, 0.3, -0.16, 0.09], Bond(i, j)))
    check_gpos_part(system, part, 1e-9)
    check_vtens_part(system, part, 1e-7)


def test_gpos_vtens_cross_water32():
    system = get_system_water32()
    part = ForcePartValence(system)
    for j in xrange(system.natom):
        if len(system.neighs1[j])==2:
            i, k = system.neighs1[j]
            part.add_term(Cross(
                    1.2,
                    1.7,
                    1.9,
                    Bond(i, j),
                    Bond(j, k),
            ))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-7)


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
                types = [system.ffatypes[i0], system.ffatypes[i1], system.ffatypes[i2], system.ffatypes[i3]]
                if types in forbidden_dihedrals or types[::-1] in forbidden_dihedrals: continue
                idih += 1
                fc = 2.1 + 0.01*(0.3*i1 + 0.7*i2)
                part = ForcePartValence(system)
                part.add_term(PolyFour([-2.0*fc,0.0001,0.0,0.0],DihedCos(i0,i1,i2,i3)))
                check_gpos_part(system, part, 1e-9)
                check_vtens_part(system, part, 1e-9)


def test_gpos_vtens_ub_water():
    system = get_system_water32()
    part = ForcePartValence(system)
    for i1 in xrange(system.natom):
        for i0 in system.neighs1[i1]:
            for i2 in system.neighs1[i1]:
                if i0 > i2:
                    part.add_term(Harmonic(2.1,2.0*angstrom,UreyBradley(i0, i1, i2)))
    check_gpos_part(system, part, 1e-10)
    check_vtens_part(system, part, 1e-10)
