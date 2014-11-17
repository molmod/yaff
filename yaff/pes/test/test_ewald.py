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

from yaff import *

from yaff.test.common import get_system_water32, get_system_quartz
from yaff.pes.test.common import check_gpos_part, check_vtens_part


def test_ewald_water32():
    # These are the energy contributions that one should get:
    # alpha           REAL            RECI            CORR
    # 0.05  -7.2121617e-01   2.4919194e-08  -1.4028899e-03
    # 0.10  -7.1142179e-01   1.5853109e-04  -1.1355778e-02
    # 0.20  -6.3429485e-01   6.1021373e-03  -9.4426328e-02
    # 0.50  -7.7158542e-02   9.3260624e-01  -1.5780667e+00
    # 1.00  -1.9121203e-05   8.2930717e+00  -9.0156717e+00
    system = get_system_water32()
    check_alpha_depedence(system)
    check_dielectric(system)


def test_ewald_quartz():
    # These are the energy contributions that one should get:
    # alpha           REAL            RECI            CORR
    # 0.05  -3.5696637e-02   6.8222205e-29  -2.5814534e-01
    # 0.10   1.3948043e-01   2.2254505e-08  -4.3332242e-01
    # 0.20   1.7482393e-01   2.9254105e-03  -4.7159132e-01
    # 0.50   5.6286111e-04   7.7869316e-01  -1.0730980e+00
    # 1.00   2.6807119e-12   4.6913539e+00  -4.9851959e+00
    system = get_system_quartz().supercell(2, 2, 2)
    check_alpha_depedence(system)

def test_ewald_dd_quartz():
    system = get_system_quartz().supercell(2, 2, 2)
    system.radii = np.random.rand( system.natom)
    system.radii2 = np.random.rand( system.natom)
    dipoles = np.random.rand( system.natom, 3 )
    system.dipoles = dipoles
    check_alpha_dependence_dd(system)


def check_alpha_depedence(system):
    # Idea: run ewald sum with two different alpha parameters and compare.
    # (this only works if both real and reciprocal part properly converge.)
    energies = []
    gposs = []
    vtenss = []
    assert abs(system.charges.sum()) < 1e-10
    for alpha in 0.05, 0.1, 0.2, 0.5, 1.0:
        energy, gpos, vtens = get_electrostatic_energy(alpha, system)
        energies.append(energy)
        gposs.append(gpos)
        vtenss.append(vtens)
    energies = np.array(energies)
    gposs = np.array(gposs)
    vtenss = np.array(vtenss)
    print energies
    assert abs(energies - energies.mean()).max() < 1e-8
    assert abs(gposs - gposs.mean(axis=0)).max() < 1e-8
    assert abs(vtenss - vtenss.mean(axis=0)).max() < 1e-8


def check_dielectric(system):
    # Idea: Using a relative permittivity epsilon (!=1) should give the same
    # results as epsilon==1 with all charges scaled by 1.0/sqrt(epsilon)
    # Initialize
    original_charges = system.charges.copy()
    energies = []
    gposs = []
    vtenss = []
    dielectric = 1.44
    # Use scaled charges and epsilon=1
    system.charges = original_charges/np.sqrt(dielectric)
    energy, gpos, vtens = get_electrostatic_energy(0.2, system)
    energies.append(energy)
    gposs.append(gpos)
    vtenss.append(vtens)
    # Use original charges and epsilon=dielectric
    system.charges = original_charges
    energy, gpos, vtens = get_electrostatic_energy(0.2, system, dielectric=dielectric)
    energies.append(energy)
    gposs.append(gpos)
    vtenss.append(vtens)
    energies = np.array(energies)
    gposs = np.array(gposs)
    vtenss = np.array(vtenss)
    assert abs(energies - energies.mean()).max() < 1e-8
    assert abs(gposs - gposs.mean(axis=0)).max() < 1e-8
    assert abs(vtenss - vtenss.mean(axis=0)).max() < 1e-8


def get_electrostatic_energy(alpha, system, dielectric=1.0):
    # Create tools needed to evaluate the energy
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.0, 0.5)
    # Construct the ewald real-space potential and part
    ewald_real_pot = PairPotEI(system.charges, alpha, rcut=5.5/alpha, dielectric=dielectric)
    part_pair_ewald_real = ForcePartPair(system, nlist, scalings, ewald_real_pot)
    assert part_pair_ewald_real.pair_pot.alpha == alpha
    # Construct the ewald reciprocal and correction part
    part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=2.0*alpha, dielectric=dielectric)
    assert part_ewald_reci.alpha == alpha
    part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings, dielectric=dielectric)
    assert part_ewald_corr.alpha == alpha
    # Construct the force field
    ff = ForceField(system, [part_pair_ewald_real, part_ewald_reci, part_ewald_corr], nlist)
    ff.update_pos(system.pos)
    ff.update_rvecs(system.cell.rvecs)
    gpos = np.zeros(system.pos.shape, float)
    vtens = np.zeros((3, 3), float)
    energy = ff.compute(gpos, vtens)
    print '    # %4.2f' % alpha, ' '.join('%15.7e' % part.energy for part in ff.parts)
    return energy, gpos, vtens


def check_alpha_dependence_dd(system):
    # Idea: run ewald sum with two different alpha parameters and compare.
    # (this only works if both real and reciprocal part properly converge.)
    energies = []
    gposs = []
    vtenss = []
    assert abs(system.charges.sum()) < 1e-10
    for alpha in 0.05, 0.1, 0.2, 0.5, 1.0:
        energy, gpos, vtens = get_electrostatic_energy_dd(alpha, system)
        energies.append(energy)
        gposs.append(gpos)
        vtenss.append(vtens)
    energies = np.array(energies)
    gposs = np.array(gposs)
    vtenss = np.array(vtenss)
    print energies
    assert abs(energies - energies.mean()).max() < 1e-8
    assert abs(gposs - gposs.mean(axis=0)).max() < 1e-8
    assert abs(vtenss - vtenss.mean(axis=0)).max() < 1e-8


def get_electrostatic_energy_dd(alpha, system):
    poltens_i = np.tile( np.diag([0.0,0.0,0.0]) , np.array([system.natom, 1]) )
    # Create tools needed to evaluate the energy
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.8, 1.0, 1.0)
    # Construct the ewald real-space potential and part
    ewald_real_pot = PairPotEIDip(system.charges, system.dipoles, poltens_i, alpha, rcut=5.5/alpha)
    part_pair_ewald_real = ForcePartPair(system, nlist, scalings, ewald_real_pot)
    assert part_pair_ewald_real.pair_pot.alpha == alpha
    # Construct the ewald reciprocal and correction part
    part_ewald_reci = ForcePartEwaldReciprocalDD(system, alpha, gcut=2.0*alpha)
    assert part_ewald_reci.alpha == alpha
    part_ewald_corr = ForcePartEwaldCorrectionDD(system, alpha, scalings)
    assert part_ewald_corr.alpha == alpha
    # Construct the force field
    ff = ForceField(system, [part_pair_ewald_real, part_ewald_reci, part_ewald_corr], nlist)
    ff.update_pos(system.pos)
    ff.update_rvecs(system.cell.rvecs)
    gpos = np.zeros(system.pos.shape, float)
    vtens = np.zeros((3, 3), float)
    energy = ff.compute(gpos, vtens)
    print '    # %4.2f' % alpha, ' '.join('%15.7e' % part.energy for part in ff.parts)
    return energy, gpos, vtens


def test_ewald_gpos_vtens_reci_water32():
    system = get_system_water32()
    dielectric = 1.4
    for alpha in 0.05, 0.1, 0.2:
        part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=alpha/0.75, dielectric=dielectric)
        check_gpos_part(system, part_ewald_reci)
        check_vtens_part(system, part_ewald_reci)


def test_ewald_gpos_vtens_reci_quartz():
    system = get_system_quartz()
    for alpha in 0.1, 0.2, 0.5:
        part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=alpha/0.5)
        check_gpos_part(system, part_ewald_reci)
        check_vtens_part(system, part_ewald_reci)


def test_ewald_reci_volchange_quartz():
    system = get_system_quartz()
    dielectric = 1.2
    for alpha in 0.1, 0.2, 0.5:
        part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=alpha/0.5, dielectric=dielectric)
        # compute the energy
        energy1 = part_ewald_reci.compute()
        # distort the cell and restore to the original volume
        volume = system.cell.volume
        reduced = np.dot(system.pos, system.cell.gvecs.transpose())
        new_rvecs = system.cell.rvecs * np.random.uniform(0.9, 1.0)
        new_volume = np.linalg.det(new_rvecs)
        new_rvecs *= (volume/new_volume)**(1.0/3.0)
        system.pos[:] = np.dot(reduced, new_rvecs)
        system.cell.update_rvecs(new_rvecs)
        # recompute the energy
        energy2 = part_ewald_reci.compute()
        # energies must be the same
        assert abs(energy1 - energy2) < 1e-5*abs(energy1)


def test_ewald_corr_quartz():
    from scipy.special import erf
    system = get_system_quartz().supercell(2, 2, 2)
    for alpha in 0.05, 0.1, 0.2:
        scalings = Scalings(system, np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings)
        energy1 = part_ewald_corr.compute()
        # self-interaction corrections
        energy2 = -alpha/np.sqrt(np.pi)*(system.charges**2).sum()
        # corrections from scaled interactions
        for i0, i1, scale, nbond in scalings.stab:
            delta = system.pos[i0] - system.pos[i1]
            system.cell.mic(delta)
            d = np.linalg.norm(delta)
            term = erf(alpha*d)/d*(1-scale)*system.charges[i0]*system.charges[i1]
            energy2 -= term
        assert abs(energy1 - energy2) < 1e-10


def test_ewald_gpos_vtens_corr_water32():
    system = get_system_water32()
    scalings = Scalings(system, 0.0, 0.0, 0.5)
    for alpha in 0.05, 0.1, 0.2:
        part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings, dielectric=0.8)
        check_gpos_part(system, part_ewald_corr)
        check_vtens_part(system, part_ewald_corr)


def test_ewald_gpos_vtens_corr_quartz():
    system = get_system_quartz().supercell(2, 2, 2)
    scalings = Scalings(system, np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
    for alpha in 0.1, 0.2, 0.5:
        part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings)
        check_gpos_part(system, part_ewald_corr)
        check_vtens_part(system, part_ewald_corr)


def test_ewald_vtens_neut_water32():
    # fake water model, negative oxygens and neutral hydrogens
    system = get_system_water32()
    system.charges -= 0.1
    for alpha in 0.05, 0.1, 0.2:
        part_ewald_neut = ForcePartEwaldNeutralizing(system, alpha)
        check_vtens_part(system, part_ewald_neut)
