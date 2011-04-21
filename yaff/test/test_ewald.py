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

from yaff import *

from common import get_system_water32, get_system_quartz, check_gpos_part, \
    check_vtens_part


def test_ewald_water32():
    system = get_system_water32()
    charges = -0.8 + (system.numbers == 1)*1.2
    check_alpha_depedence(system, charges)


def test_ewald_quartz():
    system = get_system_quartz()
    charges = 1.8 - (system.numbers == 8)*2.7
    check_alpha_depedence(system, charges)


def check_alpha_depedence(system, charges):
    # Idea: run ewald sum with two different alpha parameters and compare.
    # (this only works if both real and reciprocal part properly converge.)
    energies = []
    gposs = []
    vtenss = []
    assert abs(charges.sum()) < 1e-10
    for alpha in 0.05, 0.1, 0.2, 0.5, 1.0:
        energy, gpos, vtens = get_electrostatic_energy(alpha, system, charges)
        energies.append(energy)
        gposs.append(gpos)
        vtenss.append(vtens)
    energies = np.array(energies)
    gposs = np.array(gposs)
    vtenss = np.array(vtenss)
    assert abs(energies - energies.mean()).max() < 1e-8
    assert abs(gposs - gposs.mean(axis=0)).max() < 1e-8
    assert abs(vtenss - vtenss.mean(axis=0)).max() < 1e-8


def get_electrostatic_energy(alpha, system, charges):
    # Creat system
    nlists = NeighborLists(system)
    scalings = Scalings(system.topology, 0.0, 0.0, 0.5)
    # Construct the ewald real-space potential and part
    cutoff = 5.5/alpha
    ewald_real_pot = PairPotEI(charges, alpha, cutoff)
    ewald_real_part = PairPart(nlists, scalings, ewald_real_pot)
    # Construct the ewald reciprocal and correction part
    gmax = np.ceil(alpha*2.0/system.gspacings-0.5).astype(int)
    ewald_reci_part = EwaldReciprocalPart(system, charges, alpha, gmax)
    ewald_corr_part = EwaldCorrectionPart(system, charges, alpha, scalings)
    # Construct the force field
    ff = SumForceField(system, [ewald_real_part, ewald_reci_part, ewald_corr_part], nlists)
    ff.update_pos(system.pos)
    gpos = np.zeros(system.pos.shape, float)
    vtens = np.zeros((3, 3), float)
    return ff.compute(gpos, vtens), gpos, vtens


def test_ewald_gpos_vtens_reci_water32():
    system = get_system_water32()
    charges = -0.8 + (system.numbers == 1)*1.2
    for alpha, eps in (0.05, 1e-17), (0.1, 1e-13), (0.2, 1e-11):
        gmax = np.ceil(alpha*1.5/system.gspacings-0.5).astype(int)
        ewald_reci_part = EwaldReciprocalPart(system, charges, alpha, gmax)
        check_gpos_part(system, ewald_reci_part, eps)
        check_vtens_part(system, ewald_reci_part, eps)


def test_ewald_gpos_vtens_reci_quartz():
    system = get_system_quartz()
    charges = 1.8 - (system.numbers == 8)*2.7
    for alpha, eps in (0.1, 1e-16), (0.2, 1e-12), (0.5, 1e-12):
        gmax = np.ceil(alpha*2.0/system.gspacings-0.5).astype(int)
        ewald_reci_part = EwaldReciprocalPart(system, charges, alpha, gmax)
        check_gpos_part(system, ewald_reci_part, eps)
        check_vtens_part(system, ewald_reci_part, eps)


def test_ewald_reci_volchange_quartz():
    system = get_system_quartz()
    charges = 1.8 - (system.numbers == 8)*2.7
    for alpha in 0.1, 0.2, 0.5:
        gmax = np.ceil(alpha*2.0/system.gspacings-0.5).astype(int)
        ewald_reci_part = EwaldReciprocalPart(system, charges, alpha, gmax)
        # compute the energy
        energy1 = ewald_reci_part.compute()
        # distort the cell and restore to the original volume
        volume = system.volume
        reduced = np.dot(system.pos, system.gvecs.transpose())
        new_rvecs = system.rvecs * np.random.uniform(0.9, 1.0)
        new_volume = np.linalg.det(new_rvecs)
        new_rvecs *= (volume/new_volume)**(1.0/3.0)
        system.pos[:] = np.dot(reduced, new_rvecs)
        system.update_rvecs(new_rvecs)
        # recompute the energy
        energy2 = ewald_reci_part.compute()
        # energies must be the same
        assert abs(energy1 - energy2) < 1e-5*abs(energy1)


def test_ewald_gpos_vtens_corr_water32():
    system = get_system_water32()
    charges = -0.8 + (system.numbers == 1)*1.2
    scalings = Scalings(system.topology, 0.0, 0.0, 0.5)
    for alpha, eps in (0.05, 1e-15), (0.1, 1e-15), (0.2, 1e-12):
        gmax = np.ceil(alpha*1.5/system.gspacings-0.5).astype(int)
        ewald_corr_part = EwaldCorrectionPart(system, charges, alpha, scalings)
        check_gpos_part(system, ewald_corr_part, eps)
        check_vtens_part(system, ewald_corr_part, eps)


def test_ewald_gpos_vtens_corr_quartz():
    system = get_system_quartz()
    charges = 1.8 - (system.numbers == 8)*2.7
    scalings = Scalings(system.topology, 0.0, 0.0, 0.5)
    for alpha, eps in (0.1, 1e-12), (0.2, 1e-11), (0.5, 1e-11):
        gmax = np.ceil(alpha*2.0/system.gspacings-0.5).astype(int)
        ewald_corr_part = EwaldCorrectionPart(system, charges, alpha, scalings)
        check_gpos_part(system, ewald_corr_part, eps)
        check_vtens_part(system, ewald_corr_part, eps)


def test_ewald_vtens_neut_water32():
    # fake water model, negative oxygens and neutral hydrogens
    system = get_system_water32()
    charges = -(system.numbers == 8)*0.8
    for alpha, eps in (0.05, 1e-10), (0.1, 1e-10), (0.2, 1e-10):
        ewald_neut_part = EwaldNeutralizingPart(system, charges, alpha)
        check_vtens_part(system, ewald_neut_part, eps)
