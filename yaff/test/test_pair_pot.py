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
from scipy.special import erfc

from molmod import angstrom, kcalmol
from common import get_system_water32, get_system_caffeine

from yaff import *


def test_pair_pot_lj_water32_9A():
    # Initialize system, topology and scaling
    system = get_system_water32()
    nlists = NeighborLists(system)
    scalings = Scalings(system.topology)
    # Initialize parameters
    rminhalf_table = {1: 0.2245*angstrom, 8: 1.7682*angstrom}
    epsilon_table = {1: -0.0460*kcalmol, 8: -0.1521*kcalmol}
    sigmas = np.zeros(96, float)
    epsilons = np.zeros(96, float)
    for i in xrange(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Create the pair_pot and pair_term
    pair_pot = PairPotLJ(sigmas, epsilons, 9*angstrom)
    pair_term = PairTerm(nlists, scalings, pair_pot)
    nlists.update() # update the neighborlists, once the cutoffs are known.
    # Compute the energy using yaff.
    energy = pair_term.energy()
    # Compute the energy manually
    check_energy = 0.0
    for i in 0,:#xrange(system.natom):
        # compute the distances in the neighborlist manually and check.
        for j in xrange(i, system.natom):
            delta = system.pos[i] - system.pos[j]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            assert abs(delta).max() < 0.5*9.865*angstrom
            for l0 in xrange(-1, 2):
                for l1 in xrange(-1, 2):
                    for l2 in xrange(-1, 2):
                        if l0==0 and l1==0 and l2==0:
                            if i==j:
                                continue
                            # find the scaling
                            fac = 1.0
                            for k, s in scalings[j]:
                                if k == i:
                                    fac = s
                                    break
                            # continue if scaled to zero
                            if fac == 0.0:
                                continue
                        else:
                            # Interactions with neighboring cells are counted
                            # half. (The energy per unit cell is computed.)
                            fac = 0.5
                        my_delta = delta + np.array([l0,l1,l2])*9.865*angstrom
                        d = np.linalg.norm(my_delta)
                        if d <= nlists.cutoff:
                            sigma = 0.5*(sigmas[i]+sigmas[j])
                            epsilon = np.sqrt(epsilons[i]*epsilons[j])
                            x = (sigma/d)**6
                            term = 4*fac*epsilon*(x*(x-1))
                            check_energy += term
    assert abs(energy - check_energy) < 1e-15


def test_pair_pot_lj_caffeine_15A():
    # Get a system and define scalings
    system = get_system_caffeine()
    scalings = Scalings(system.topology, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    rminhalf_table = {
        1: 0.2245*angstrom,
        6: 1.6000*angstrom,
        7: 1.7000*angstrom,
        8: 1.7682*angstrom
    }
    epsilon_table = {
        1: -0.0460*kcalmol,
        6: -0.2357*kcalmol,
        7: -0.1970*kcalmol,
        8: -0.1521*kcalmol,
    }
    sigmas = np.zeros(96, float)
    epsilons = np.zeros(96, float)
    for i in xrange(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Construct the pair potential
    pair_pot = PairPotLJ(sigmas, epsilons, 15*angstrom)
    # The pair function
    def pair_fn(i, j, d):
        sigma = 0.5*(sigmas[i]+sigmas[j])
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)**6
        return 4*epsilon*(x*(x-1))

    check_pair_pot_caffeine(system, scalings, pair_pot, pair_fn, 1e-15)


def test_pair_pot_ei1_caffeine_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    scalings = Scalings(system.topology, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    charges = np.random.uniform(0, 1, system.natom)
    charges -= charges.sum()
    # Construct the pair potential
    cutoff = 10*angstrom
    alpha = 3.5/cutoff
    pair_pot = PairPotEI(charges, alpha, cutoff)
    # The pair function
    def pair_fn(i, j, d):
        return charges[i]*charges[j]*erfc(alpha*d)/d

    check_pair_pot_caffeine(system, scalings, pair_pot, pair_fn, 1e-13)


def test_pair_pot_ei2_caffeine_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    scalings = Scalings(system.topology, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    charges = np.random.uniform(0, 1, system.natom)
    charges -= charges.sum()
    # Construct the pair potential
    cutoff = 10*angstrom
    alpha = 0.0
    pair_pot = PairPotEI(charges, alpha, cutoff)
    # The pair function
    def pair_fn(i, j, d):
        return charges[i]*charges[j]/d

    check_pair_pot_caffeine(system, scalings, pair_pot, pair_fn, 1e-13)


def check_pair_pot_caffeine(system, scalings, pair_pot, pair_fn, eps):
    nlists = NeighborLists(system)
    # Create the pair_term
    pair_term = PairTerm(nlists, scalings, pair_pot)
    nlists.update() # update the neighborlists, once the cutoffs are known.
    # Compute the energy using yaff.
    energy = pair_term.energy()
    # Compute the energy manually
    check_energy = 0.0
    for i in 0,:#xrange(system.natom):
        # compute the distances in the neighborlist manually and check.
        for j in xrange(i+1, system.natom):
            delta = system.pos[i] - system.pos[j]
            # find the scaling
            fac = 1.0
            for k, s in scalings[j]:
                if k == i:
                    fac = s
                    break
            # continue if scaled to zero
            if fac == 0.0:
                continue
            d = np.linalg.norm(delta)
            if d <= nlists.cutoff:
                check_energy += fac*pair_fn(i, j, d)
    assert abs(energy - check_energy) < 1e-13


def test_gradient_pair_pot_ei2_caffeine_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    scalings = Scalings(system.topology, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    charges = np.random.uniform(0, 1, system.natom)
    charges -= charges.sum()
    # Construct the pair potential
    cutoff = 30*angstrom
    alpha = 0.0
    pair_pot = PairPotEI(charges, alpha, cutoff)
    # Create the FF term
    nlists = NeighborLists(system)
    pair_term = PairTerm(nlists, scalings, pair_pot)
    # Compute the energy using yaff.
    nlists.update()
    energy1, gradient1 = pair_term.energy_gradient()
    # Change the positions, and recompute
    delta = np.random.normal(0, 1e-4, (system.natom, 3))
    system.pos += delta
    nlists.update()
    energy2, gradient2 = pair_term.energy_gradient()
    # Check
    first = energy2 - energy1
    second = 0.5*np.dot(delta.ravel(), (gradient1 + gradient2).ravel())
    print first, second
    assert abs(first - second) < abs(first)*1e-3
