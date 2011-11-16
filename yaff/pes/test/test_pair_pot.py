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

from yaff.test.common import get_system_water32, get_system_caffeine, \
    get_system_2atoms
from yaff.pes.test.common import check_gpos_part, check_vtens_part

from yaff import *



#
# Auxiliary function
#


def get_scaling(scalings, srow, a, b):
    stab = scalings.stab
    if srow >= len(stab):
        return srow, 1.0
    while stab['a'][srow] < a:
        srow += 1
        if srow >= len(stab):
            return srow, 1.0
    while stab['b'][srow] < b and stab['a'][srow] == a:
        srow += 1
        if srow >= len(stab):
            return srow, 1.0
    if stab['a'][srow] == a and stab['b'][srow] == b:
        return srow, stab['scale'][srow]
    return srow, 1.0


#
# Water tests
#


def check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, eps):
    # Update the neighborlists, once the rcuts are known.
    nlist.update()
    # Compute the energy using yaff.
    energy1 = part_pair.compute()
    gpos = np.zeros(system.pos.shape, float)
    energy2 = part_pair.compute(gpos)
    # Compute in python as a double check
    srow = 0
    check_energy = 0.0
    for a in xrange(system.natom):
        # compute the distances in the neighborlist manually and check.
        for b in xrange(system.natom):
            delta = system.pos[b] - system.pos[a]
            delta -= np.floor(delta/(9.865*angstrom)+0.5)*(9.865*angstrom)
            assert abs(delta).max() < 0.5*9.865*angstrom
            for r2 in xrange(0, 2):
                for r1 in xrange((r2!=0)*-1, 2):
                    for r0 in xrange((r2!=0 or r1!=0)*-1, 2):
                        if r0==0 and r1==0 and r2==0:
                            if a<=b:
                                continue
                            # find the scaling
                            srow, fac = get_scaling(scalings, srow, a, b)
                            assert a > b
                            # continue if scaled to zero
                            if fac == 0.0:
                                continue
                        else:
                            fac = 1.0
                        my_delta = delta + np.array([r0,r1,r2])*9.865*angstrom
                        d = np.linalg.norm(my_delta)
                        if d <= nlist.rcut:
                            check_energy += fac*pair_fn(a, b, d)
    #print "energy1 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy1, check_energy, energy1-check_energy)
    #print "energy2 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy2, check_energy, energy2-check_energy)
    assert abs(energy1 - check_energy) < eps
    assert abs(energy2 - check_energy) < eps


def get_part_water32_9A_lj():
    # Initialize system, nlist and scaling
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system)
    # Initialize parameters
    rminhalf_table = {1: 0.2245*angstrom, 8: 1.7682*angstrom}
    epsilon_table = {1: -0.0460*kcalmol, 8: -0.1521*kcalmol}
    sigmas = np.zeros(96, float)
    epsilons = np.zeros(96, float)
    for i in xrange(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Create the pair_pot and part_pair
    rcut = 9*angstrom
    pair_pot = PairPotLJ(sigmas, epsilons, rcut, Hammer(1.0))
    assert abs(pair_pot.sigmas - sigmas).max() == 0.0
    assert abs(pair_pot.epsilons - epsilons).max() == 0.0
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Create a pair function:
    def pair_fn(i, j, d):
        sigma = 0.5*(sigmas[i]+sigmas[j])
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)**6
        return 4*epsilon*(x*(x-1))*np.exp(1.0/(d-rcut))
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_lj_water32_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_lj()
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_water32_9A_mm3():
    # Initialize system, nlist and scaling
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system)
    # Initialize parameters
    sigma_table  = {1: 1.62*angstrom, 8: 1.82*angstrom}
    epsilon_table = {1: 0.020*kcalmol, 8: 0.059*kcalmol}
    sigmas = np.zeros(96, float)
    epsilons = np.zeros(96, float)
    for i in xrange(system.natom):
        sigmas[i] = sigma_table[system.numbers[i]]
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Create the pair_pot and part_pair
    rcut = 9*angstrom
    pair_pot = PairPotMM3(sigmas, epsilons, rcut, Hammer(1.0))
    assert abs(pair_pot.sigmas - sigmas).max() == 0.0
    assert abs(pair_pot.epsilons - epsilons).max() == 0.0
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Create a pair function:
    def pair_fn(i, j, d):
        sigma = 0.5*(sigmas[i]+sigmas[j])
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)
        if d<rcut:
            return epsilon*(1.84e5*np.exp(-12.0/x)-2.25*x**6)*np.exp(1.0/(d-rcut))
        else:
            return 0.0
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_mm3_water32_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_mm3()
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_water32_9A_grimme():
    # Initialize system, nlist and scaling
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system)
    # Initialize parameters
    r0_table = {1: 1.001*angstrom, 8: 1.342*angstrom}
    c6_table = {1: 0.14*1e-3*kjmol*nanometer**6, 8: 0.70*1e-3*kjmol*nanometer**6}
    r0s = np.zeros(96, float)
    c6s = np.zeros(96, float)
    for i in xrange(system.natom):
        r0s[i] = r0_table[system.numbers[i]]
        c6s[i] = c6_table[system.numbers[i]]
    # Create the pair_pot and part_pair
    rcut = 9*angstrom
    pair_pot = PairPotGrimme(r0s, c6s, rcut, Hammer(1.0))
    assert abs(pair_pot.r0 - r0s).max() == 0.0
    assert abs(pair_pot.c6 - c6s).max() == 0.0
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Create a pair function:
    def pair_fn(i, j, d):
        r0 = (r0s[i]+r0s[j])
        c6 = np.sqrt(c6s[i]*c6s[j])
        if d<rcut:
            return -1.1/(1.0 + np.exp(-20.0*(d/r0-1.0)))*c6/d**6*np.exp(1.0/(d-rcut))
        else:
            return 0.0
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_grimme_water32_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_grimme()
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-10)


def get_part_water32_4A_exprep(amp_mix, amp_mix_coeff, b_mix, b_mix_coeff):
    # Initialize system, nlist and scaling
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 1.0)
    # Initialize parameters
    amps = np.array([2.3514195495e+00, 4.2117588157e+02])
    bs = np.array([4.4107388814e+00, 4.4661933834e+00])/angstrom
    # Allocate some arrays for the pair potential
    assert len(system.ffatypes) == 2
    amp_cross = np.zeros((2, 2), float)
    b_cross = np.zeros((2, 2), float)
    # Create the pair_pot and part_pair
    rcut = 4*angstrom
    pair_pot = PairPotExpRep(
        system.ffatype_ids, amp_cross, b_cross, rcut, Switch3(2.0),
        amps, amp_mix, amp_mix_coeff, bs, b_mix, b_mix_coeff,
    )
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Create a pair function:
    def pair_fn(i0, i1, d):
        amp0 = amps[system.ffatype_ids[i0]]
        amp1 = amps[system.ffatype_ids[i1]]
        b0 = bs[system.ffatype_ids[i0]]
        b1 = bs[system.ffatype_ids[i1]]
        if amp_mix == 0:
            amp = np.sqrt(amp0*amp1)
        elif amp_mix == 1:
            cor = 1-amp_mix_coeff*abs(np.log(amp0/amp1))
            amp = np.exp( (np.log(amp0)+np.log(amp1))/2*cor)
        else:
            raise NotImplementedError
        if b_mix == 0:
            b = (b0+b1)/2
        elif b_mix == 1:
            cor = 1-b_mix_coeff*abs(np.log(amp0/amp1))
            b = (b0+b1)/2*cor
        else:
            raise NotImplementedError
        # truncation
        if d > rcut - 2.0:
            x = (rcut - d)/2.0
            amp *= (3-2*x)*x*x
        energy = amp*np.exp(-b*d)
        return energy
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_exprep_water32_4A_case1():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_4A_exprep(0, 0.0, 0, 0.0)
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-12)


def test_pair_pot_exprep_water32_4A_case2():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_4A_exprep(1, 2.385e-2, 1, 7.897e-3)
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-12)



#
# Caffeine tests
#


def check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, eps):
    nlist.update() # update the neighborlists, once the rcuts are known.
    # Compute the energy using yaff.
    energy1 = part_pair.compute()
    gpos = np.zeros(system.pos.shape, float)
    energy2 = part_pair.compute(gpos)
    # Compute the energy manually
    check_energy = 0.0
    srow = 0
    for a in xrange(system.natom):
        # compute the distances in the neighborlist manually and check.
        for b in xrange(a):
            delta = system.pos[b] - system.pos[a]
            # find the scaling
            srow, fac = get_scaling(scalings, srow, a, b)
            # continue if scaled to zero
            if fac == 0.0:
                continue
            d = np.linalg.norm(delta)
            if d < nlist.rcut:
                energy = fac*pair_fn(a, b, d)
                #print 'P', a, b, d/angstrom, fac, energy
                check_energy += energy
    print "energy1 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy1, check_energy, energy1-check_energy)
    print "energy2 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy2, check_energy, energy2-check_energy)
    assert abs(energy1 - check_energy) < eps
    assert abs(energy2 - check_energy) < eps



def get_part_caffeine_lj_15A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
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
    sigmas = np.zeros(24, float)
    epsilons = np.zeros(24, float)
    for i in xrange(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Construct the pair potential and part
    pair_pot = PairPotLJ(sigmas, epsilons, 15*angstrom)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        sigma = 0.5*(sigmas[i]+sigmas[j])
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)**6
        return 4*epsilon*(x*(x-1))
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_lj_caffeine_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_lj_15A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-10)


def get_part_caffeine_mm3_15A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
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
    sigmas = np.zeros(24, float)
    epsilons = np.zeros(24, float)
    for i in xrange(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Construct the pair potential and part
    pair_pot = PairPotMM3(sigmas, epsilons, 15*angstrom)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        sigma = 0.5*(sigmas[i]+sigmas[j])
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)
        return epsilon*(1.84e5*np.exp(-12.0/x)-2.25*x**6)
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_mm3_caffeine_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_mm3_15A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_caffeine_grimme_15A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    r0_table = {
        1: 1.001*angstrom,
        6: 1.452*angstrom,
        7: 1.397*angstrom,
        8: 1.342*angstrom,
    }
    c6_table = {
        1: 0.14*1e-3*kjmol*nanometer**6,
        6: 1.75*1e-3*kjmol*nanometer**6,
        7: 1.23*1e-3*kjmol*nanometer**6,
        8: 0.70*1e-3*kjmol*nanometer**6,
    }
    r0s = np.zeros(24, float)
    c6s = np.zeros(24, float)
    for i in xrange(system.natom):
        r0s[i] = r0_table[system.numbers[i]]
        c6s[i] = c6_table[system.numbers[i]]
    # Construct the pair potential and part
    pair_pot = PairPotGrimme(r0s, c6s, 15*angstrom)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        r0 = (r0s[i]+r0s[j])
        c6 = np.sqrt(c6s[i]*c6s[j])
        return -1.1/(1.0 + np.exp(-20.0*(d/r0-1.0)))*c6/d**6
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_grimme_caffeine_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_grimme_15A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_caffeine_exprep_5A(amp_mix, amp_mix_coeff, b_mix, b_mix_coeff):
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 1.0)
    # Initialize (random) parameters
    amps = np.array([2.35, 410.0, 0.0, 421.0])
    bs = np.array([4.46, 4.43, 0.0, 4.41])/angstrom
    # Allocate some arrays for the pair potential
    assert len(system.ffatypes) == 4
    amp_cross = np.zeros((4, 4), float)
    b_cross = np.zeros((4, 4), float)
    # Construct the pair potential and part
    pair_pot = PairPotExpRep(
        system.ffatype_ids, amp_cross, b_cross, 5*angstrom, None,
        amps, amp_mix, amp_mix_coeff, bs, b_mix, b_mix_coeff,
    )
    assert abs(np.diag(pair_pot.amp_cross) - amps).max() < 1e-10
    assert abs(np.diag(pair_pot.b_cross) - bs).max() < 1e-10
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i0, i1, d):
        amp0 = amps[system.ffatype_ids[i0]]
        amp1 = amps[system.ffatype_ids[i1]]
        b0 = bs[system.ffatype_ids[i0]]
        b1 = bs[system.ffatype_ids[i1]]
        if amp_mix == 0:
            amp = np.sqrt(amp0*amp1)
        elif amp0 == 0.0 or amp1 == 0.0:
            amp = 0.0
        elif amp_mix == 1:
            cor = 1-amp_mix_coeff*abs(np.log(amp0/amp1))
            amp = np.exp( (np.log(amp0)+np.log(amp1))/2*cor )
        else:
            raise NotImplementedError
        if b_mix == 0:
            b = (b0+b1)/2
        elif amp0 == 0.0 or amp1 == 0.0:
            b = 0.0
        elif b_mix == 1:
            cor = 1-b_mix_coeff*abs(np.log(amp0/amp1))
            b = (b0+b1)/2*cor
        else:
            raise NotImplementedError
        if amp == 0.0 or b == 0.0:
            energy = 0.0
        else:
            energy = amp*np.exp(-b*d)
        return energy
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_exprep_caffeine_5A_case1():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_exprep_5A(0, 0.0, 0, 0.0)
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def test_pair_pot_exprep_caffeine_5A_case2():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_exprep_5A(1, 2.385e-2, 1, 7.897e-3)
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_caffeine_dampdisp_9A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 1.0)
    # Initialize (very random) parameters
    c6s = np.array([2.5, 27.0, 18.0, 13.0])
    bs = np.array([2.5, 2.0, 0.0, 1.8])
    vols = np.array([5, 3, 4, 5])*angstrom**3
    # Allocate some arrays
    assert system.nffatype == 4
    c6_cross = np.zeros((4, 4), float)
    b_cross = np.zeros((4, 4), float)
    # Construct the pair potential and part
    pair_pot = PairPotDampDisp(system.ffatype_ids, c6_cross, b_cross, 9*angstrom, None, c6s, bs, vols)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i0, i1, d):
        c60 = c6s[system.ffatype_ids[i0]]
        c61 = c6s[system.ffatype_ids[i1]]
        b0 = bs[system.ffatype_ids[i0]]
        b1 = bs[system.ffatype_ids[i1]]
        vol0 = vols[system.ffatype_ids[i0]]
        vol1 = vols[system.ffatype_ids[i1]]
        ratio = (vol0/vol1)**2
        c6 = 2*c60*c61/(c60*ratio+c61/ratio)
        if b0 != 0 and b1 != 0:
            b = 0.5*(b0+b1)
            damp = 0
            fac = 1
            for k in xrange(7):
                damp += (b*d)**k/fac
                fac *= k+1
            damp = 1 - np.exp(-b*d)*damp
            return -c6/d**6*damp
        else:
            damp = 1
            return -c6/d**6
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_dampdisp_caffeine_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_dampdisp_9A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_caffeine_ei1_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    system.charges = np.random.uniform(0, 1, system.natom)
    system.charges -= system.charges.sum()
    # Construct the pair potential and part
    rcut = 10*angstrom
    alpha = 3.5/rcut
    pair_pot = PairPotEI(system.charges, alpha, rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        return system.charges[i]*system.charges[j]*erfc(alpha*d)/d
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_ei1_caffeine_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei1_10A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-9)


def get_part_caffeine_ei2_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    system.charges = np.random.uniform(0, 1, system.natom)
    system.charges -= system.charges.sum()
    # Construct the pair potential and part
    rcut = 10*angstrom
    alpha = 0.0
    pair_pot = PairPotEI(system.charges, alpha, rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        return system.charges[i]*system.charges[j]*erfc(alpha*d)/d
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_ei2_caffeine_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei2_10A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-8)


#
# Water derivative tests
#


def test_gpos_vtens_pair_pot_water_lj_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_lj()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


#
# Caffeine derivative tests
#


def test_gpos_vtens_pair_pot_caffeine_lj_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_lj_15A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_mm3_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_mm3_15A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_grimme_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_grimme_15A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_exprep_5A_case1():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_exprep_5A(0, 0.0, 0, 0.0)
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_exprep_5A_case2():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_exprep_5A(1, 2.385e-2, 1, 7.897e-3)
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_dampdisp_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_dampdisp_9A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_ei1_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei1_10A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_caffeine_ei2_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei2_10A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


#
# Tests for special cases
#


def test_pair_pot_grimme_2atoms():
    system = get_system_2atoms()
    nlist = NeighborList(system)
    scalings = Scalings(system, 1.0, 1.0, 1.0)
    R0 = 1.452*angstrom
    C6 = 1.75*1e-3*kjmol*nanometer**6
    pair_pot = PairPotGrimme(np.array([R0, R0]), np.array([C6, C6]), 15*angstrom)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    nlist.update()
    energy = part_pair.compute()
    d = np.sqrt(sum((system.pos[0]-system.pos[1])**2))
    check_energy=-1.1/( 1.0 + np.exp(-20.0*(d/(2.0*R0)-1.0)) )*C6/d**6
    #print "d=%f" %d
    #print "energy=%f , check_energy=%f" %(energy, check_energy)
    assert abs(energy-check_energy)<1e-10
