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
'''Polarizable forcefields'''
#TODO: in which subpackage does this belong?

import numpy as np

from yaff.log import log
from yaff.sampling import Hook
from yaff.pes import Scalings
from molmod.units import angstrom, kjmol, debye

__all__ = ['RelaxDipoles','DipolSCPicard','DipolRules','get_ei_tensors']

class RelaxDipoles(Hook):
    def __init__(self, start=0, step=1):
        """
           **Arguments:**

           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        #Implement determination of dipoles here.
        #Atomic positions and charges are available as iterative.ff.system.pos and
        #iterative.ff.system.charges
        #The dipoles can be set through iterative.ff.part_pair_eidip.pair_pot.dipoles

        #Check there is a pair_pot for dipoles present in the forcefield
        part_names = [part.name for part in iterative.ff.parts]
        assert 'pair_eidip' in part_names, "ff has to contain pair_eidip when using dipoles"
        #Get array containing polarizability tensors
        poltens_i = iterative.ff.part_pair_eidip.pair_pot.poltens_i

        #Compute the dipoles
        #newdipoles = DipolSCPicard( iterative.ff.system.pos, iterative.ff.system.charges, poltens_i,
        #                                    iterative.ff.system.natom, system=iterative.ff.system)
        newdipoles = DipolRules( iterative.ff.system.pos, iterative.ff.system.charges,
                                            iterative.ff.system.natom, system=iterative.ff.system)

        #Set the new dipoles
        iterative.ff.part_pair_eidip.pair_pot.dipoles = newdipoles



def DipolRules( pos,charges,natom,system):
    dipoles = np.zeros ( (natom,3) )
    for i, atom in enumerate(system.numbers):
        #Oxygen atom
        if atom == 8:
            #Find neighbours, there should be two
            neighs = list(system.neighs1[i])
            assert len(neighs)==2, "O atom with index %d should have 2 neighbours!"%i
            #Find interatomic vectors with neighbours
            R_01 = system.pos[i,:] - system.pos[neighs[0],:]
            d_01 = np.linalg.norm(R_01)
            R_01 /= d_01
            R_02 = system.pos[i,:] - system.pos[neighs[1],:]
            d_02 = np.linalg.norm(R_02)
            R_02 /= d_02
            results =  [[-0.2803971,  -0.23086147,  0.31158304]]
            A_coef = results[0][0]*d_01 + results[0][1]*(charges[i] - charges[neighs[0]]) + results[0][2]
            B_coef = results[0][0]*d_02 + results[0][1]*(charges[i] - charges[neighs[1]]) + results[0][2]
            dipoles[i,:] = A_coef*R_01 + B_coef*R_02
        elif atom==1:
            #Find neighbours, there should be two
            neighs = list(system.neighs1[i])
            assert len(neighs)==1, "H atom with index %d should have 2 neighbours!"%i
            #Find interatomic vectors with neighbours
            R_01 = system.pos[i,:] - system.pos[neighs[0],:]
            d_01 = np.linalg.norm(R_01)
            R_01 /= d_01
            dipoles[i,:] = 0.077*R_01
    return dipoles






def DipolSCPicard(pos, charges, poltens_i, natom, init=None, conv_crit=1e-10, tensors=None, system=None):
    """
    Determine point dipoles that minimize the electrostatic energy using self-
    consistent method with Picard update (following Wang-Skeel 2005).
    Initial guess is zero
    Only for non-periodic systems!
    No scalings are applied!
    Right now focus on code readability, not on speed.
    TODO: Put this step in c code?

        **Arguments:**

        pos
            Atomic positions (also the positions of point dipoles)

        charges
            Atomic point charges at atomic positions

       poltens_i
            Tensor that gives the inverse of atomic polarizabilities (3natom x 3atom )
            Creation of point dipoles contributes to the electrostatic energy,
            this creation energy depends on the atomic polarizabilities.


       **Optional arguments:**

       init
            Initial guess for the dipoles
        conv_crit
            The self-consistent method is considered to be converged as soon
            as the RMSD of a dipole update falls between this value.

    """
    #Get tensors that describe electrostatic energy
    if tensors is None:
        G_0, G_1, G_2, D, chi = get_ei_tensors( pos, poltens_i, natom, system)
    else:
        G_0 = tensors['G_0']
        G_1 = tensors['G_1']
        G_2 = tensors['G_2']
        D   = tensors['D']
        chi = tensors['chi']
    #d contains the current guess for the dipoles in following form:
    # [d0x,d0y,d0z,d1x,d1y,d1z,...]
    if init is not None:
        d = init #Set or compute initial guess
    else:
        d = np.ones( (3*natom,) )

    converged = False
    steps = 0
    if log.do_high:
        log.hline()
        log('Starting Picard algorithm to determine dipoles')
        log('   Step     RMSD')
        log.hline()
    #Take Picard steps until convergence is achieved
    while (not converged and steps<20):
        d_new = - np.dot( D , ( np.dot(G_1,charges) + np.dot(G_2,d) + chi    ) )
        #Compute the RMSD between new and old dipoles
        rmsd_dipoles = np.sqrt(((d-d_new)**2).mean())
        if log.do_high:
            log('%7i %20.15f' % (steps, rmsd_dipoles) )
        if rmsd_dipoles < conv_crit:
            converged = True
        d = d_new
        steps += 1
    if log.do_high:
        log.hline()
        log('Picard algorithm converged after %d steps'%steps)
        log.hline()



    #Reshape dipoles to [natom x 3] matrix
    dipoles = np.reshape( d , (natom,3) )
    return dipoles


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


def get_ei_tensors( pos, poltens_i, natom, system, scalings=[1.0,1.0,1.0], A_1=-1.0/angstrom**2,A_2=0.0*kjmol/debye ):
    """
    Compute tensors that help in evaluating electrostatic energy of charges and
    dipoles.
    """
    scalings = Scalings(system, scalings[0], scalings[1], scalings[2])

    #Construct tensors, notation from Wang-Skeel 2005
    #G_0,  a [natom x natom] matrix describing the interaction between point charges and point charges
    G_0 = np.zeros( (natom,natom) )
    srow = 0
    #Loop over first charge
    for i in xrange(natom):
        #Loop over second charge
        for j in xrange(natom):
            if j==i: continue #Exclude self-interaction
            srow, fac = get_scaling(scalings, srow, i,j)
            delta = pos[i,:] - pos[j,:]     #Interatomic distance vector
            r = np.linalg.norm( delta )  #Interatomic distance
            G_0[i,j] = 1.0/r*fac

    #G_1, a [3*natom x natom] matrix describing the interaction between point charges and point dipoles
    G_1 = np.zeros( (3*natom,natom) )
    srow = 0
    #Loop over dipoles
    for i in xrange(natom):
        #Loop over charges
        for j in xrange(natom):
            if j==i: continue #Exclude self-interaction
            srow, fac = get_scaling(scalings, srow, i,j)
            delta = pos[i,:] - pos[j,:]     #Interatomic distance vector
            r = np.linalg.norm( delta )  #Interatomic distance
            r3 = 1.0/r**3
            #Loop over x,y and z component of dipole
            for k in xrange(3):
                G_1[3*i+k,j] = delta[k]*r3*fac

    #G_2, a [3*natom x 3*natom] matrix describing the interaction between point dipoles and point dipoles
    G_2 = np.zeros( (3*natom,3*natom) )
    srow = 0
    #First loop over dipoles
    for i in xrange(natom):
        #Second loop over dipoles
        for j in xrange(natom):
            if j==i: continue #Exclude self-interaction
            srow, fac = get_scaling(scalings, srow, i,j)
            delta = pos[i,:] - pos[j,:]     #Interatomic distance vector
            r = np.linalg.norm( delta )  #Interatomic distance
            r3 = 1.0/r**3
            r5 = 1.0/r**5
            #Loop over x,y and z component of first dipole
            for k in xrange(3):
                #Loop over x,y and z component of second dipole
                for l in xrange(3):
                    G_2[3*i+k,3*j+l] = -3.0*delta[k]*delta[l]*r5*fac
                    if k==l: G_2[3*i+k,3*j+l] += r3*fac

    #D, a [3*natom x 3*natom] matrix describing the atomic polarizabilities.
    #This will be a block diagonal matrix if polarizabilities are not coupled.
    #We still need to invert the given poltens_i
    D = np.linalg.inv(poltens_i)
    #chi a [3natom,] vector that enable to impose direction on a dipole, for example
    #keep an oxygen dipole on bisector between two Si atoms
    chi = np.zeros( (3*natom,) )
    #Loop over atoms
    for i in xrange(natom):
        if system.numbers[i]==8: #Do oxygen atoms
            #Find neighbours, there should be two
            neighs = list(system.neighs1[i])
            assert len(neighs)==2, "O atom with index %d should have 2 neighbours!"%i
            #Find interatomic vectors with neighbours
            R_01 = system.pos[i,:] - system.pos[neighs[0],:]
            R_02 = system.pos[i,:] - system.pos[neighs[1],:]
            #Normal to these two vectors:
            n = np.cross( R_01, R_02 )
            n /= np.linalg.norm(n)
            for k in xrange(3):
                #Lower energy when dipole aligns with charge weighted bisector
                chi[3*i+k] += A_1* ( system.charges[neighs[0]] - system.charges[i] ) * R_01[k]/np.linalg.norm(R_01)
                chi[3*i+k] += A_1* ( system.charges[neighs[1]] - system.charges[i] ) * R_02[k]/np.linalg.norm(R_02)
                #Raise energy when dipole goes out of plane
                chi[3*i+k] += A_2* n[k]
        if system.numbers[i]==6:     #Do carbon atoms
            #Find neighbours, there should be four
            neighs = list(system.neighs1[i])
            assert len(neighs)==4, "C atom with index %d should have 2 neighbours!"%i
            #Find neighbour that is not hydrogen
            for n in xrange(len(neighs)):
                if system.numbers[neighs[n]]!=1: neigh = neighs[n]
            R_01 = system.pos[i,:] - system.pos[neigh,:]
            for k in xrange(3):
                chi[3*i+k] += A_1 * system.charges[neigh]*R_01[k]/np.linalg.norm(R_01)





    return G_0, G_1, G_2, D, chi
