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


import pkg_resources
import numpy as np
from scipy.special import erfc, erf
from nose.tools import assert_raises

from molmod import angstrom, kcalmol

from yaff.test.common import get_system_water32, get_system_caffeine, \
    get_system_2atoms, get_system_quartz, get_system_water, \
    get_system_4113_01WaterWater
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


def check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, eps, rmax=1):
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
            for r2 in xrange(0, rmax+1):
                for r1 in xrange((r2!=0)*(-rmax), rmax+1):
                    for r0 in xrange((r2!=0 or r1!=0)*-(rmax), rmax+1):
                        if r0==0 and r1==0 and r2==0:
                            if a<=b:
                                continue
                            # find the scaling
                            srow, fac = get_scaling(scalings, srow, a, b)
                            # continue if scaled to zero
                            if fac == 0.0:
                                continue
                        else:
                            fac = 1.0
                        my_delta = delta + np.array([r0,r1,r2])*9.865*angstrom
                        d = np.linalg.norm(my_delta)
                        if d <= nlist.rcut:
                            my_energy = fac*pair_fn(a, b, d, my_delta)
                            #print 'P %3i %3i (% 3i % 3i % 3i) %10.7f %3.1f %10.3e' % (a, b, r0, r1, r2, d, fac, my_energy)
                            check_energy += my_energy
    print "energy1 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy1, check_energy, energy1-check_energy)
    print "energy2 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy2, check_energy, energy2-check_energy)
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
    def pair_fn(i, j, d, delta):
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
    onlypaulis = np.zeros(96, np.int32)
    for i in xrange(system.natom):
        sigmas[i] = sigma_table[system.numbers[i]]
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Create the pair_pot and part_pair
    rcut = 9*angstrom
    pair_pot = PairPotMM3(sigmas, epsilons, onlypaulis, rcut, Hammer(1.0))
    assert abs(pair_pot.sigmas - sigmas).max() == 0.0
    assert abs(pair_pot.epsilons - epsilons).max() == 0.0
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Create a pair function:
    def pair_fn(i, j, d, delta):
        sigma = sigmas[i]+sigmas[j]
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)
        if d<rcut:
            return epsilon*(1.84e5*np.exp(-12.0/x)-2.25*x**6)*np.exp(1.0/(d-rcut))
        else:
            return 0.0
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_mm3_water32_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_mm3()
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-12)


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
    def pair_fn(i, j, d, delta):
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
    def pair_fn(i0, i1, d, delta):
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


def get_part_water32_14A_ei(radii=None):
    # Initialize system, nlist and scaling
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.5, 1.0)
    dielectric = 1.0
    # Create the pair_pot and part_pair
    rcut = 14*angstrom
    alpha = 5.5/rcut
    pair_pot = PairPotEI(system.charges, alpha, rcut, dielectric=dielectric, radii=radii)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d, delta):
        r_ij = np.sqrt( pair_pot.radii[i]**2 + pair_pot.radii[j]**2 )
        if r_ij == 0.0: pot = 1.0
        else: pot = erf(d/r_ij)
        return system.charges[i]*system.charges[j]*(pot-erf(alpha*d))/d
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_ei_water32_14A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_14A_ei()
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-12, rmax=1)


def test_pair_pot_ei_water32_14A_gaussiancharges():
    radii = np.array( [1.50, 1.20, 1.20]*32 ) * angstrom
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_14A_ei(radii=radii)
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-12, rmax=1)


def get_part_water32_14A_eidip():
    # Initialize system, nlist and scaling
    system = get_system_water32()
    #Reset charges
    system.charges *= 0.0
    #Set dipoles to random values
    dipoles = np.random.rand( system.natom, 3 )
    #Set polarizations to infinity (no energy to create dipoles)
    poltens_i = np.tile( np.diag([0.0,0.0,0.0]) , np.array([system.natom, 1]) )
    nlist = NeighborList(system)

    scalings = Scalings(system, 1.0, 1.0, 1.0)
    # Create the pair_pot and part_pair
    rcut = 14*angstrom
    alpha = 5.5/rcut
    pair_pot = PairPotEIDip(system.charges, dipoles, poltens_i, alpha, rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    part_pair.nlist.update()
    # The pair function
    def pair_fn(i, j, d, delta):
        energy = 0.0
        #Dipole-Dipole (only term for this test)
        fac1 = erfc(alpha*d) + 2.0*alpha*d/np.sqrt(np.pi)*np.exp(-alpha**2*d**2)
        fac2 = 3.0*erfc(alpha*d) + 4.0*alpha**3*d**3/np.sqrt(np.pi)*np.exp(-alpha**2*d**2) \
                + 6.0*alpha*d/np.sqrt(np.pi)*np.exp(-alpha**2*d**2)
        energy += np.dot( pair_pot.dipoles[i,:] , pair_pot.dipoles[j,:] )*fac1/d**3 - \
                         1.0*np.dot(pair_pot.dipoles[i,:],delta)*np.dot(delta,pair_pot.dipoles[j,:])*fac2/d**5
        return energy
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_eidip_water32_14A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_14A_eidip()
    check_pair_pot_water32(system, nlist, scalings, part_pair, pair_fn, 1e-12, rmax=1)


def get_part_water_eidip(scalings = [0.5,1.0,1.0],rcut=14.0*angstrom,switch_width=0.0*angstrom, finite=False, alpha=0.0, do_radii=False):
    '''
    Make a system with one water molecule with a point dipole on every atom,
    setup a ForcePart...
    '''
    # Set dipoles
    dipoles = np.array( [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0 ]] ) # natom x 3
    # Initialize system, nlist and scaling
    system = get_system_water()
    #TODO make radii2 system attribute
    system.radii = np.array( [ 1.5,1.2,1.2] ) * angstrom
    system.radii2 = np.array( [1.6,1.3,1.2] ) * angstrom
    if finite:
        #Make a system with point dipoles approximated by two charges
        system = make_system_finite_dipoles(system, dipoles, eps=0.0001*angstrom)
    if not do_radii:
        system.radii *= 0.0
        if not finite:system.radii2 *= 0.0
    nlist = NeighborList(system)
    scalings = Scalings(system, scalings[0], scalings[1], scalings[2])
    # Set poltens
    poltens_i = np.tile( 0.0*np.diag([1.0,1.0,1.0]) , np.array([system.natom, 1]) )
    # Create the pair_pot and part_pair
    if finite:
        pair_pot = PairPotEI(system.charges,alpha, rcut, tr=Switch3(switch_width), radii=system.radii)
    else:
        pair_pot = PairPotEIDip(system.charges, dipoles, poltens_i, alpha, rcut, tr=Switch3(switch_width), radii=system.radii, radii2=system.radii2)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    nlist.update()
    #Make a different nlist in case we approximate the point dipoles with charges
    #Interactions between charges at the same site should be excluded
    if finite:
        neigh_dtype = [
        ('a', int), ('b', int), ('d', float),        # a & b are atom indexes, d is the distance
        ('dx', float), ('dy', float), ('dz', float), # relative vector (includes cell vectors of image cell)
        ('r0', int), ('r1', int), ('r2', int)        # position of image cell.
            ]
        nneigh = np.sum( nlist.neighs[0:nlist.nneigh]['d'] > 0.2*angstrom )
        new_neighs = np.zeros(nneigh, dtype=neigh_dtype)
        counter = 0
        for n in nlist.neighs[0:nlist.nneigh]:
            if n['d']>0.2*angstrom:
                new_neighs[counter] = n
                counter += 1
        nlist.neighs = new_neighs
        nlist.nneigh = nneigh
    # The pair function
    def pair_fn(i, j, d, delta):
        energy = 0.0
        #Charge-Charge
        energy += system.charges[i]*system.charges[j]/d
        if not finite:
            #Charge-Dipole
            energy += system.charges[i]*np.dot(delta, pair_pot.dipoles[j,:])/d**3
            #Dipole-Charge
            energy -= system.charges[j]*np.dot(delta, pair_pot.dipoles[i,:])/d**3
            #Dipole-Dipole
            energy += np.dot( pair_pot.dipoles[i,:] , pair_pot.dipoles[j,:] )/d**3 - \
                             3*np.dot(pair_pot.dipoles[i,:],delta)*np.dot(delta,pair_pot.dipoles[j,:])/d**5
        if d > rcut - switch_width:
            x = (rcut - d)/switch_width
            energy *= (3-2*x)*x*x
        return energy
    return system, nlist, scalings, part_pair, pair_pot, pair_fn


def test_pair_pot_eidip_water_finite():
    #Compare the electrostatic energy of a system with point dipoles with the
    #energy of a system with these dipoles approximated by two point charges
    #TODO: what happens to the virial tensor in this case?
    #TODO: handle scalings in this test
    for do_radii in True, False:
        for alpha in 0.0, 2.0:
            #Get the electrostatic energy of a water molecule with atomic point dipoles approximated by two charges
            system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip(scalings=[1.0,1.0,1.0],finite=True,alpha=alpha, do_radii=do_radii)
            gpos1 = np.zeros(system.pos.shape, float)
            energy1 = part_pair.compute(gpos1)
            #Reshape gpos1
            gpos1 = np.asarray([ np.sum( gpos1[i::3], axis=0 ) for i in xrange(system.natom/3)])
            #Get the electrostatic energy of a water molecule with atomic point dipoles
            system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip(scalings=[1.0,1.0,1.0],finite=False,alpha=alpha, do_radii=do_radii)
            gpos2 = np.zeros(system.pos.shape, float)
            energy2 = part_pair.compute(gpos2)
            #Finite difference approximation is not very accurate...
            assert np.abs(energy1 - energy2) < 1.0e-5
            assert abs(gpos1 - gpos2).max() < 1e-5


def check_pair_pot_water(system, nlist, scalings, part_pair, pair_pot, pair_fn, eps):

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
                energy = fac*pair_fn(a, b, d, delta)
                check_energy += energy
    #Add dipole creation energy
    check_energy += 0.5*np.dot( np.transpose(np.reshape( pair_pot.dipoles, (-1,) )) , np.dot( pair_pot.poltens_i, np.reshape( pair_pot.dipoles, (-1,) ) ) )
    print "energy1 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy1, check_energy, energy1-check_energy)
    print "energy2 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy2, check_energy, energy2-check_energy)
    assert abs(energy1 - check_energy) < eps
    assert abs(energy2 - check_energy) < eps


def test_pair_pot_eidip_water_setdipoles():
    # '''Test if we can modify dipoles of PairPotEIDip object'''
    #Setup simple system
    system = get_system_water()
    rcut = 50.0*angstrom
    #Some arrays representing dipoles
    dipoles0 = np.array( [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0 ]] ) # natom x 3
    dipoles1 = np.array( [[9.0,8.0,7.0],[6.0,5.0,4.0],[3.0,2.0,1.0 ]] ) # natom x 3
    dipoles2 = np.array( [[9.0,8.0],[6.0,5.0],[3.0,2.0 ]] ) # natom x 2
    dipoles3 = np.array( [[9.0,8.0,7.0],[6.0,5.0,4.0]] ) # natom x 2
    #Array representing atomic polarizability tensors
    poltens_i = np.tile( np.diag([1.0,1.0,1.0]) , np.array([system.natom, 1]) )
    system.dipoles = dipoles0
    #Initialize pair potential
    pair_pot = PairPotEIDip(system.charges, system.dipoles, poltens_i, 0.0, rcut)
    #Check if dipoles are initialized correctly
    assert np.all( pair_pot.dipoles == dipoles0 )
    #Update the dipoles to new values
    system.dipoles[:] = dipoles1
    assert np.all( pair_pot.dipoles == dipoles1 )
    #Try to update dipoles of PairPot directly, this should raise an attribute error
    with assert_raises(AttributeError):
        pair_pot.dipoles = dipoles1


def test_pair_pot_eidip_water():
    #Setup system and force part
    system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip()
    #Check energy from Yaff with manually computed energy
    check_pair_pot_water(system, nlist, scalings, part_pair, pair_pot, pair_fn, 1.0e-12)
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist, symm_vtens=False)
    #Again, but with a truncation scheme (truncation scheme settings are ridiculous,
    #but this is needed to see an effect for water)

    #Setup system and force part
    system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip(rcut=2.0*angstrom,switch_width=1.5*angstrom)
    #Check energy from Yaff with manually computed energy
    check_pair_pot_water(system, nlist, scalings, part_pair, pair_pot, pair_fn, 1.0e-12)
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist, symm_vtens=False)


def make_system_finite_dipoles(system, dipoles, eps=0.05*angstrom):
    '''
    Make a system where point dipoles are replaced by finite dipoles consisting
    of two charges separated by eps with charges +|d|/eps and -|d|/eps.
    Special attention has to be paid to the nlist, as we do not want to include
    interactions between charges around the same atom.
    '''
    ncharges = 3 #Original charge + two charges to approximate dipole
    newsystem = {}
    #Copy some 'easy' attributes of the orginal system
    #No repetitions
    newsystem['ffatypes'] = system.ffatypes
    newsystem['bonds'] = system.bonds #No new connections are introduced
    #Three repetitions
    newsystem['numbers'] = np.tile( system.numbers, ncharges)
    if system.radii is None and system.radii2 is None: newsystem['radii'] = None
    else:
        if system.radii is None: system.radii = np.zeros( (natom,1) )
        if system.radii2 is None: system.radii2 = np.zeros( (natom,1) )
    newsystem['radii'] = np.reshape( np.asarray([system.radii,system.radii2, system.radii2]), (-1,) )
    newsystem['masses'] = np.tile( system.masses, ncharges)
    newsystem['ffatype_ids'] = np.tile( system.ffatype_ids, ncharges)
    #Cell vectors
    newsystem['rvecs'] = system.cell.rvecs
    #Charges
    d_norms = np.sqrt( np.sum( dipoles**2 , axis = 1 ) )
    ac = np.zeros( system.natom*ncharges )
    ac[0*system.natom:1*system.natom] = system.charges
    ac[1*system.natom:2*system.natom] = d_norms/eps*0.5
    ac[2*system.natom:3*system.natom] = -d_norms/eps*0.5
    newsystem['charges'] = ac
    #Positions
    d_norms[d_norms==0.0] = 1.0
    pos = np.zeros( (system.natom*ncharges,3 ))
    pos[0*system.natom:1*system.natom,:] = system.pos
    pos[1*system.natom:2*system.natom,:] = system.pos - eps*dipoles/np.transpose(np.reshape( np.tile(d_norms,3), (3,-1) ))
    pos[2*system.natom:3*system.natom,:] = system.pos + eps*dipoles/np.transpose(np.reshape( np.tile(d_norms,3), (3,-1) ))
    return System(
        numbers=newsystem['numbers'],
        pos=pos,
        ffatypes=newsystem['ffatypes'],
        ffatype_ids=newsystem['ffatype_ids'],
        bonds=newsystem['bonds'],
        rvecs=newsystem['rvecs'],
        charges=newsystem['charges'],
        radii=newsystem['radii'],
        masses=newsystem['masses'] )


def test_pair_pot_ei_water32_dielectric():
    #Using a relative permittivity epsilon (!=1) should give the same results
    #as epsilon==1 with all charges scaled by 1.0/sqrt(epsilon)
    # Initialize system, nlist, scaling, ...
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.5, 1.0)
    rcut = 14*angstrom
    alpha = 5.5/rcut
    dielectric = 1.44
    #Compute energy with epsilon 1 and scaled charges
    pair_pot = PairPotEI(system.charges/np.sqrt(dielectric), alpha, rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    ff = ForceField(system, [part_pair], nlist)
    ff.update_pos(system.pos)
    gpos0 = np.zeros(system.pos.shape, float)
    vtens0 = np.zeros((3, 3), float)
    energy0 = ff.compute(gpos0, vtens0)
    #Compute energy with epsilon=dielctric and original charges
    pair_pot = PairPotEI(system.charges, alpha, rcut, dielectric=dielectric)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    ff = ForceField(system, [part_pair], nlist)
    ff.update_pos(system.pos)
    gpos1 = np.zeros(system.pos.shape, float)
    vtens1 = np.zeros((3, 3), float)
    energy1 = ff.compute(gpos1, vtens1)
    assert np.abs(energy0-energy1) < 1.0e-10
    assert np.all(np.abs(gpos0-gpos1)) < 1.0e-10
    assert np.all(np.abs(vtens0-vtens1) < 1.0e-10 )


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
    onlypaulis = np.zeros(24, np.int32)
    for i in xrange(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Construct the pair potential and part
    pair_pot = PairPotMM3(sigmas, epsilons, onlypaulis, 15*angstrom)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        sigma = sigmas[i]+sigmas[j]
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)
        return epsilon*(1.84e5*np.exp(-12.0/x)-2.25*x**6)
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_mm3_caffeine_15A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_mm3_15A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-12)


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


def get_part_caffeine_ljcross_9A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.0, 1.0)
    # Initialize (very random) parameters
    assert system.nffatype == 4
    eps_cross = np.array([[1.0,3.5,4.6,9.4],
                          [3.5,2.0,4.4,4.1],
                          [4.6,4.4,5.0,3.3],
                          [9.4,4.1,3.3,0.1]])
    sig_cross = np.array([[0.2,0.5,1.6,1.4],
                          [0.5,1.0,2.4,1.1],
                          [1.6,2.4,1.0,2.3],
                          [1.4,1.1,2.3,0.9]])
    assert np.all( np.abs( sig_cross - np.transpose(sig_cross) ) < 1e-15 )
    assert np.all( np.abs( eps_cross - np.transpose(eps_cross) ) < 1e-15 )
    # Construct the pair potential and part
    pair_pot = PairPotLJCross(system.ffatype_ids, eps_cross, sig_cross, 9*angstrom)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i0, i1, d):
        ffat0 = system.ffatype_ids[i0]
        ffat1 = system.ffatype_ids[i1]
        epsilon = eps_cross[ffat0,ffat1]
        sigma = sig_cross[ffat0,ffat1]
        E = 4.0*epsilon*( (sigma/d)**12.0 - (sigma/d)**6.0 )
        #print "%2d %2d %4.1f %+7.4f" % (i0,i1,d,E)
        return E
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_ljcross_caffeine_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ljcross_9A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_caffeine_dampdisp_9A(power=6):
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 1.0)
    # Initialize (very random) parameters
    cns = np.array([2.5, 27.0, 18.0, 13.0])
    bs = np.array([2.5, 2.0, 0.0, 1.8])
    vols = np.array([5, 3, 4, 5])*angstrom**3
    # Allocate some arrays
    assert system.nffatype == 4
    cn_cross = np.zeros((4, 4), float)
    b_cross = np.zeros((4, 4), float)
    # Construct the pair potential and part
    pair_pot = PairPotDampDisp(system.ffatype_ids, cn_cross, b_cross, 9*angstrom, None, cns, bs, vols, power=power)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i0, i1, d):
        cn0 = cns[system.ffatype_ids[i0]]
        cn1 = cns[system.ffatype_ids[i1]]
        b0 = bs[system.ffatype_ids[i0]]
        b1 = bs[system.ffatype_ids[i1]]
        vol0 = vols[system.ffatype_ids[i0]]
        vol1 = vols[system.ffatype_ids[i1]]
        ratio = vol0/vol1
        cn = 2*cn0*cn1/(cn0/ratio+cn1*ratio)
        if b0 != 0 and b1 != 0:
            b = 0.5*(b0+b1)
            damp = 0
            fac = 1
            for k in xrange(power+1):
                damp += (b*d)**k/fac
                fac *= k+1
            damp = 1 - np.exp(-b*d)*damp
            return -cn/d**power*damp
        else:
            damp = 1
            return -cn/d**power
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_dampdisp_caffeine_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_dampdisp_9A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def test_pair_pot_dampdisp8_caffeine_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_dampdisp_9A(power=8)
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def test_pair_pot_dampdisp10_caffeine_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_dampdisp_9A(power=10)
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-15)


def get_part_caffeine_ei1_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
    dielectric = 1.0
    # Initialize (random) parameters
    system.charges = np.random.uniform(0, 1, system.natom)
    system.charges -= system.charges.sum()
    # Construct the pair potential and part
    rcut = 10*angstrom
    alpha = 3.5/rcut
    pair_pot = PairPotEI(system.charges, alpha, rcut, dielectric=dielectric)
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
    dielectric = 1.0
    # Initialize (random) parameters
    system.charges = np.random.uniform(0, 1, system.natom)
    system.charges -= system.charges.sum()
    # Construct the pair potential and part
    rcut = 10*angstrom
    alpha = 0.0
    pair_pot = PairPotEI(system.charges, alpha, rcut, dielectric=dielectric)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        return system.charges[i]*system.charges[j]*erfc(alpha*d)/d
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_ei2_caffeine_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei2_10A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-8)


def get_part_caffeine_ei3_10A():
    # Get a system and define scalings
    system = get_system_caffeine()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 0.5)
    # Initialize (random) parameters
    system.charges = np.random.uniform(0, 1, system.natom)
    system.charges -= system.charges.sum()
    #Set the atomic radii
    radii = np.random.uniform(0,1,system.natom)
    # Construct the pair potential and part
    rcut = 10*angstrom
    alpha = 0.0
    pair_pot = PairPotEI(system.charges, alpha, rcut, radii=radii)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # The pair function
    def pair_fn(i, j, d):
        r_ij = np.sqrt( pair_pot.radii[i]**2 + pair_pot.radii[j]**2 )
        return system.charges[i]*system.charges[j]*erf(d/r_ij)/d
    return system, nlist, scalings, part_pair, pair_fn


def test_pair_pot_ei3_caffeine_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei3_10A()
    check_pair_pot_caffeine(system, nlist, scalings, part_pair, pair_fn, 1e-9)


#
# 4113_01WaterWater tests
#


def get_part_4113_01WaterWater_eislater1s1scorr():
    # Get a system and define scalings
    system = get_system_4113_01WaterWater()
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 1.0)
    rcut = 20*angstrom
    pair_pot = PairPotEiSlater1s1sCorr(system.radii, system.valence_charges, system.charges - system.valence_charges, rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    def pair_fn(i, j, R, alpha, beta):
        E = 0.0
        delta = beta-alpha
        if alpha==0.0 or beta==0.0:
            if beta!=0.0:
                E = -np.exp(-R/beta)/R - 0.5/beta*np.exp(-R/beta)
            elif alpha!=0.0:
                E = -np.exp(-R/alpha)/R - 0.5/alpha*np.exp(-R/alpha)
        elif np.abs(delta)<0.025:
            alphaR = R/alpha
            T0 = -np.exp(-alphaR)/R - np.exp(-alphaR)/48.0/R*(alphaR**3 + 9.0*alphaR**2 + 33.0*alphaR)
            T1 = -1.0/96.0/alpha**2*( alphaR**3 + 6.0*alphaR**2 + 15.0*alphaR + 15.0)*np.exp(-alphaR)
            T2 = -1.0/480.0/alpha**3*( 3.0*alphaR**4 + 5.0*alphaR**3 - 15.0*alphaR**2 - 60.0*alphaR - 60.0)*np.exp(-alphaR)
            E = T0 + T1*delta + T2*delta**2/2.0
        else:
            E -= ( beta**3*np.exp(-R/beta) + alpha**3*np.exp(-R/alpha) ) / (2.0*(alpha**2-beta**2)**2)
            E -= ( beta**4*(3.0*alpha**2-beta**2)*np.exp(-R/beta) - alpha**4*(3.0*beta**2-alpha**2)*np.exp(-R/alpha) ) / (R*(alpha**2-beta**2)**3)
        return E
    return system, nlist, scalings, part_pair, pair_fn


def get_part_4113_01WaterWater_olpslater1s1s():
    # Get a system and define scalings
    system = get_system_4113_01WaterWater()
    #system = system.subsystem([2,3])
    #print system.radii
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 1.0, 1.0)
    rcut = 20*angstrom
    # Define some parameters for the exchange term
    ex_scale = 1.0
    corr_a = 16.0
    corr_b = 2.4
    corr_c = -0.2
    # Make the pair potential
    pair_pot = PairPotOlpSlater1s1s(system.radii, system.valence_charges, ex_scale, rcut, corr_a=corr_a, corr_b=corr_b, corr_c=corr_c)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    def pair_fn(i, j, R, alpha, beta):
        E = 0.0
        delta = beta-alpha
        if np.abs(delta)<0.025:
            alphaR = R/alpha
            T0 = (alphaR**2+3.0*alphaR+3.0)*np.exp(-alphaR)/192.0/np.pi/alpha**3
            T1 = (alphaR**3-2.0*alphaR**2-9.0*alphaR-9.0)*np.exp(-alphaR)/384.0/np.pi/alpha**4
            T2 = (3.0*alphaR**4-25.0*alphaR**3+5.0*alphaR**2+90.0*alphaR+90.0)*np.exp(-alphaR)/1920.0/np.pi/alpha**5
            E = T0 + T1*delta + 0.5*T2*delta**2
        else:
            E = (alpha*np.exp(-R/alpha) + beta*np.exp(-R/beta))/8.0/np.pi/(alpha-beta)**2/(alpha+beta)**2
            E += alpha**2*beta**2*(np.exp(-R/beta) - np.exp(-R/alpha))/2.0/np.pi/R/(alpha-beta)**3/(alpha+beta)**3
        return E*ex_scale*(1.0+corr_c*(system.valence_charges[i]+system.valence_charges[j]))*(1.0-np.exp(corr_a-corr_b*R/np.sqrt(alpha*beta)))
    return system, nlist, scalings, part_pair, pair_fn


def get_part_4113_01WaterWater_disp68bjdamp():
    # Get a system and define scalings
    system = get_system_4113_01WaterWater()
    #system = system.subsystem([2,3])
    #print system.radii
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.0, 1.0, 1.0)
    rcut = 20*angstrom
    # Define some parameters for the dispersion
    c6_cross = np.array([[  0.        ,   4.19523224,   4.28064173,  23.14022933,   4.20534285,    4.2056618 ],
                         [  4.19523224,   0.        ,   0.82846012,   4.19857256,   0.81388705,    0.81394877],
                         [  4.28064173,   0.82846012,   0.        ,   4.28405005,   0.83045673,    0.83051972],
                         [ 23.14022933,   4.19857256,   4.28405005,   0.        ,   4.20869122,    4.20901042],
                         [  4.20534285,   0.81388705,   0.83045673,   4.20869122,   0.        ,    0.81591041],
                         [  4.2056618 ,   0.81394877,   0.83051972,   4.20901042,   0.81591041,    0.        ]] )
    c8_cross = np.array([[   0.       ,    35.7352303,    36.7355119,   372.3924437 ,   35.85391291,    35.85754569],
                         [  35.7352303,     0.        ,    3.76477196,   35.77871306,    3.67442289,     3.67479519],
                         [  36.7355119,     3.76477196,    0.        ,   36.78021181,    3.77727539,     3.77765811],
                         [ 372.3924437,    35.77871306,   36.78021181,    0.        ,   35.89754009,    35.90117729],
                         [  35.85391291,    3.67442289,    3.77727539,   35.89754009,    0.        ,     3.68699979],
                         [  35.85754569,    3.67479519,    3.77765811,   35.90117729,    3.68699979,     0.        ]] )
    R_cross = np.zeros((system.natom,system.natom))
    c8_scale = 1.71910290
    bj_a = 0.818471488
    bj_b = 0.0

    # Make the pair potential
    pair_pot = PairPotDisp68BJDamp(system.ffatype_ids, c6_cross, c8_cross, R_cross, rcut, c8_scale=c8_scale,bj_a=bj_a,bj_b=bj_b)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    def pair_fn(i, j, R, alpha, beta):
        if c6_cross[i,j] != 0.0: R0 = np.sqrt(c8_cross[i,j]/c6_cross[i,j])
        else: R0 = 0.0
        E = -c6_cross[i,j]/(R**6+(bj_a*R0+bj_b)**6) - c8_scale*c8_cross[i,j]/(R**8+(bj_a*R0+bj_b)**8)
        return E
    return system, nlist, scalings, part_pair, pair_fn


def get_part_4113_01WaterWater_chargetransferslater1s1s():
    # Get a system and define scalings
    system = get_system_4113_01WaterWater()
    #system = system.subsystem([2,3])
    #print system.radii
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.0, 1.0)
    rcut = 20*angstrom
    # Define some parameters for the exchange term
    ct_scale = 0.01363842
    width_power = 3.0
    # Make the pair potential
    pair_pot = PairPotChargeTransferSlater1s1s(system.radii, system.valence_charges, ct_scale, rcut, width_power=width_power)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    def pair_fn(i, j, R, alpha, beta):
        E = 0.0
        delta = beta-alpha
        if np.abs(delta)<0.025:
            alphaR = R/alpha
            T0 = (alphaR**2+3.0*alphaR+3.0)*np.exp(-alphaR)/192.0/np.pi/alpha**3
            T1 = (alphaR**3-2.0*alphaR**2-9.0*alphaR-9.0)*np.exp(-alphaR)/384.0/np.pi/alpha**4
            T2 = (3.0*alphaR**4-25.0*alphaR**3+5.0*alphaR**2+90.0*alphaR+90.0)*np.exp(-alphaR)/1920.0/np.pi/alpha**5
            E = T0 + T1*delta + 0.5*T2*delta**2
        else:
            E = (alpha*np.exp(-R/alpha) + beta*np.exp(-R/beta))/8.0/np.pi/(alpha-beta)**2/(alpha+beta)**2
            E += alpha**2*beta**2*(np.exp(-R/beta) - np.exp(-R/alpha))/2.0/np.pi/R/(alpha-beta)**3/(alpha+beta)**3
        return -E*ct_scale/(alpha*beta)**width_power
    return system, nlist, scalings, part_pair, pair_fn


def check_pair_pot_4113_01WaterWater(system, nlist, scalings, part_pair, pair_fn, eps, do_cores=False, mult_pop=True):
    nlist.update() # update the neighborlists, once the rcuts are known.
    # Compute the energy using yaff.
    energy1 = part_pair.compute()
    gpos = np.zeros(system.pos.shape, float)
    energy2 = part_pair.compute(gpos)
    # Compute the energy manually
    check_energy = 0.0
    srow = 0
    core_charges = system.charges - system.valence_charges
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
                energy  = fac*pair_fn(a, b, d, system.radii[a], system.radii[b])
                if mult_pop:
                    energy *= system.valence_charges[a]*system.valence_charges[b]
                if do_cores:
                    energy += fac*pair_fn(a, b, d, 0.0, system.radii[b])*core_charges[a]*system.valence_charges[b]
                    energy += fac*pair_fn(a, b, d, system.radii[a], 0.0)*system.valence_charges[a]*core_charges[b]
                    energy += fac*pair_fn(a, b, d, 0.0, 0.0)*core_charges[a]*core_charges[b]
                check_energy += energy
    print "energy1 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy1, check_energy, energy1-check_energy)
    print "energy2 % 18.15f     check_energy % 18.15f     error % 18.15f" %(energy2, check_energy, energy2-check_energy)
    assert abs(energy1 - check_energy) < eps
    assert abs(energy2 - check_energy) < eps


def test_pair_pot_4113_01WaterWater_eislater1s1scorr():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_eislater1s1scorr()
    check_pair_pot_4113_01WaterWater(system, nlist, scalings, part_pair, pair_fn, 1e-8, do_cores=True)


def test_pair_pot_4113_01WaterWater_olpslater1s1s():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_olpslater1s1s()
    check_pair_pot_4113_01WaterWater(system, nlist, scalings, part_pair, pair_fn, 1e-8)


def test_pair_pot_4113_01WaterWater_disp68bjdamp():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_disp68bjdamp()
    check_pair_pot_4113_01WaterWater(system, nlist, scalings, part_pair, pair_fn, 1e-8, mult_pop=False)


def test_pair_pot_4113_01WaterWater_chargetransferslater1s1s():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_chargetransferslater1s1s()
    check_pair_pot_4113_01WaterWater(system, nlist, scalings, part_pair, pair_fn, 1e-8)


#
# Water derivative tests
#


def test_gpos_vtens_pair_pot_water_lj_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_9A_lj()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pair_pot_ei_water32_14A_gaussiancharges():
    radii = np.array( [1.50, 1.20, 1.20]*32 ) * angstrom
    system, nlist, scalings, part_pair, pair_fn = get_part_water32_14A_ei(radii=radii)
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


def test_gpos_vtens_pair_pot_caffeine_ljcross_9A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ljcross_9A()
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


def test_gpos_vtens_pair_pot_caffeine_ei2_10A():
    system, nlist, scalings, part_pair, pair_fn = get_part_caffeine_ei3_10A()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pot_4113_01WaterWater_eislater1s1scorr():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_eislater1s1scorr()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pot_4113_01WaterWater_olpslater1s1s():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_olpslater1s1s()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pot_4113_01WaterWater_disp68bjdamp():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_disp68bjdamp()
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist)


def test_gpos_vtens_pot_4113_01WaterWater_chargetransferslater1s1s():
    system, nlist, scalings, part_pair, pair_fn = get_part_4113_01WaterWater_chargetransferslater1s1s()
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


def test_bks_isfinite():
    system = get_system_quartz()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_bks.txt')
    ff = ForceField.generate(system, fn_pars)
    assert np.isfinite(ff.part_pair_dampdisp.pair_pot.cn_cross).all()
    assert np.isfinite(ff.part_pair_dampdisp.pair_pot.b_cross).all()
    ff.compute()
    assert np.isfinite(ff.part_pair_exprep.energy)
    assert np.isfinite(ff.part_pair_ei.energy)
    assert np.isfinite(ff.part_ewald_reci.energy)
    assert np.isfinite(ff.part_ewald_cor.energy)
    assert np.isfinite(ff.part_ewald_neut.energy)
    assert np.isfinite(ff.part_pair_dampdisp.energy)
    assert np.isfinite(ff.energy)


def test_bks_vtens_gpos_parts():
    system = get_system_quartz()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_bks.txt')
    ff = ForceField.generate(system, fn_pars, smooth_ei=True, reci_ei='ignore')
    for part in ff.parts:
        check_vtens_part(system, part, ff.nlist)
        check_gpos_part(system, part, ff.nlist)

#
# Tests for toy systems
#
def check_dipole_finite_difference(system, nlist, part_pair, eps):
    # Collect data about the dipoles
    a1s = part_pair.pair_pot.slater1s_widths
    a1p = part_pair.pair_pot.slater1p_widths
    N1s = part_pair.pair_pot.slater1s_N
    Z1s = part_pair.pair_pot.slater1s_Z
    N1p = part_pair.pair_pot.slater1p_N
    Z1p = part_pair.pair_pot.slater1p_Z
    # Construct a new system with at every site charges to approximate the dipoles + the original charges
    # Those damn Slater dipole are however not that easy to approximate with
    # Slater monopole, oh no. You need the recurrence relation for the Slater
    # densities, which involves a derivative to alpha and to x. This leads to
    # a finite difference approximation with 4 sites for each dipole.

    # Set finite difference interval
    delta = 0.001*angstrom
    sigma = 0.001
    pos = []
    bonds = []
    widths = []
    Ns = []
    Zs = []
    for i in xrange(system.natom):
        pos.append(system.pos[i])
        for j in xrange(2): pos.append(system.pos[i] + delta*np.array([1.0,0.0,0.0]))
        for j in xrange(2): pos.append(system.pos[i] - delta*np.array([1.0,0.0,0.0]))
        for j in xrange(2): pos.append(system.pos[i] + delta*np.array([0.0,1.0,0.0]))
        for j in xrange(2): pos.append(system.pos[i] - delta*np.array([0.0,1.0,0.0]))
        for j in xrange(2): pos.append(system.pos[i] + delta*np.array([0.0,0.0,1.0]))
        for j in xrange(2): pos.append(system.pos[i] - delta*np.array([0.0,0.0,1.0]))
        # Connect all these charges so they do not get accounted in the pair pot.
        for j in xrange(13):
            for k in xrange(j+1,13):
                bonds.append([13*i+j,13*i+k])
        widths.append(a1s[i])
        Ns.append(N1s[i])
        Zs.append(Z1s[i])
        for j in xrange(3):
            for k in xrange(2):
                widths.append(a1p[i,j]+sigma)
                widths.append(a1p[i,j]-sigma)
            Ns.append( 0.25/sigma/delta*N1p[i,j]*a1p[i,j]**-3*(a1p[i,j]+sigma)**4*0.25)
            Ns.append(-0.25/sigma/delta*N1p[i,j]*a1p[i,j]**-3*(a1p[i,j]-sigma)**4*0.25)
            Ns.append(-0.25/sigma/delta*N1p[i,j]*a1p[i,j]**-3*(a1p[i,j]+sigma)**4*0.25)
            Ns.append( 0.25/sigma/delta*N1p[i,j]*a1p[i,j]**-3*(a1p[i,j]-sigma)**4*0.25)
            Zs.append(0.5/delta*Z1p[i,j])
            Zs.append(0.0)
            Zs.append(-0.5/delta*Z1p[i,j])
            Zs.append(0.0)
    pos = np.asarray(pos)
    numbers = np.repeat(system.numbers,13)
    bonds = np.asarray(bonds)
    widths = np.asarray(widths)
    Ns = np.asarray(Ns)
    Zs = np.asarray(Zs)
    # Construct a new system
    system_fd = System(numbers,pos,bonds=bonds)
    # Construct a new pair potential
    nlist_fd = NeighborList(system_fd)
    scalings = Scalings(system_fd, 0.0, 0.0, 0.0)
    rcut = 20.0*angstrom
    pair_pot_fd = PairPotEiSlater1s1sCorr(widths,Ns,Zs,rcut)
    part_pair_fd = ForcePartPair(system_fd, nlist_fd, scalings, pair_pot_fd)
    nlist.update() # update the neighborlists, once the rcuts are known.
    nlist_fd.update()
    # Finally compare the energy of the two approaches
    energy1 = part_pair.compute()
    energy2 = part_pair_fd.compute()
    rel_err = np.abs(energy1-energy2)/energy1
    assert rel_err < eps


def test_pair_pot_eislater1sp1spcorr():
    # """Test dipole implementation by approximating dipole with monopoles"""
    # Make a toy system with just two atoms
    system = System(np.array([1,2]),np.array([[0.0,0.0,0.0],[0.4,0.7,0.5]]),bonds=np.array([]))
    nlist = NeighborList(system)
    scalings = Scalings(system, 0.0, 0.0, 0.0)
    rcut = 20.0*angstrom
    # Multipole sizes
    N1s = np.array([0.5,2.0])
    N1p = np.array([[0.9,4.0,3.0],[1.2,1.1,0.45]])
    Z1s = np.array([2.0,8.0])
    Z1p = np.array([[2.0,0.0,3.0],[3.0,4.0,06.0]])
    # Slater-widths, very different
    a1s = np.array([0.5,0.6])
    a1p = np.array([[0.5,0.5,0.5],[0.6,0.6,0.6]])
    pair_pot = PairPotEiSlater1sp1spCorr(a1s,N1s,Z1s,a1p,N1p,Z1p,rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Check with finite difference (don't expect impressive accuracy!)
    nlist.update()
    check_dipole_finite_difference(system, nlist, part_pair, 1e-4)
    # Check gradient and virial tensor
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist, symm_vtens=False)
    # Slater-widths, nearly equal
    a1s = np.array([0.5,0.501])
    a1p = np.array([[0.5,0.5,0.5],[0.501,0.501,0.501]])
    pair_pot = PairPotEiSlater1sp1spCorr(a1s,N1s,Z1s,a1p,N1p,Z1p,rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Check with finite difference (don't expect impressive accuracy!)
    nlist.update()
    check_dipole_finite_difference(system, nlist, part_pair, 1e-4)
    # Check gradient and virial tensor
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist, symm_vtens=False)
    # Slater-widths, equal
    a1s = np.array([0.5,0.5])
    a1p = np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]])
    pair_pot = PairPotEiSlater1sp1spCorr(a1s,N1s,Z1s,a1p,N1p,Z1p,rcut)
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Check with finite difference (don't expect impressive accuracy!)
    nlist.update()
    check_dipole_finite_difference(system, nlist, part_pair, 1e-4)
    # Check gradient and virial tensor
    check_gpos_part(system, part_pair, nlist)
    check_vtens_part(system, part_pair, nlist, symm_vtens=False)
