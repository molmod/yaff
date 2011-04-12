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

from common import get_system_water32, get_system_quartz, check_gradient_term


def test_ewald_water32():
    # Idea: run ewald sum with two different alpha parameters and compare.
    # (this only works if both real and reciprocal part properly converge.)
    energies = []
    system = get_system_water32()
    charges = -0.8 + (system.numbers == 1)*1.2
    assert abs(charges.sum()) < 1e-10
    for alpha in 0.05, 0.1, 0.2, 0.5, 1.0:
        energies.append(get_electrostatic_energy(alpha, system, charges))
    energies = np.array(energies)
    assert abs(energies - energies.mean()).max() < 1e-8


def test_ewald_quartz():
    # Idea: run ewald sum with two different alpha parameters and compare.
    # (this only works if both real and reciprocal part properly converge.)
    energies = []
    system = get_system_quartz()
    charges = 1.8 - (system.numbers == 8)*2.7
    assert abs(charges.sum()) < 1e-10
    for alpha in 0.05, 0.051, 0.052, 0.1, 0.2, 0.5, 1.0:
        energies.append(get_electrostatic_energy(alpha, system, charges))
    energies = np.array(energies)
    assert abs(energies - energies.mean()).max() < 1e-8


def get_electrostatic_energy(alpha, system, charges):
    # Creat system
    nlists = NeighborLists(system)
    scalings = Scalings(system.topology, 0.0, 0.0, 0.5)
    # Construct the ewald real-space potential and term
    cutoff = 5.5/alpha
    ewald_real_pot = PairPotEI(charges, alpha, cutoff)
    ewald_real_term = PairTerm(nlists, scalings, ewald_real_pot)
    # Construct the ewald reciprocal and correction term
    gmax = np.ceil(alpha*2.0/system.gspacings-0.5).astype(int)
    ewald_reci_term = EwaldReciprocalTerm(system, charges, alpha, gmax)
    ewald_corr_term = EwaldCorrectionTerm(system, charges, alpha, scalings)
    # Construct the force field
    ff = SumForceField(system, [ewald_real_term, ewald_reci_term, ewald_corr_term], nlists)
    ff.update_pos(system.pos)
    return ff.compute()


def test_ewald_gradient_reci_water32():
    system = get_system_water32()
    charges = -0.8 + (system.numbers == 1)*1.2
    for alpha, eps in (0.05, 1e-17), (0.1, 1e-13), (0.2, 1e-11):
        gmax = np.ceil(alpha*1.5/system.gspacings-0.5).astype(int)
        ewald_reci_term = EwaldReciprocalTerm(system, charges, alpha, gmax)
        check_gradient_term(system, ewald_reci_term, eps)


def test_ewald_gradient_reci_quartz():
    system = get_system_quartz()
    charges = 1.8 - (system.numbers == 8)*2.7
    for alpha, eps in (0.1, 1e-16), (0.2, 1e-12), (0.5, 1e-12):
        gmax = np.ceil(alpha*2.0/system.gspacings-0.5).astype(int)
        ewald_reci_term = EwaldReciprocalTerm(system, charges, alpha, gmax)
        check_gradient_term(system, ewald_reci_term, eps)
