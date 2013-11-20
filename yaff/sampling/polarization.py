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

__all__ = ['RelaxDipoles','DipolSCPicard','get_ei_tensors']

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
        newdipoles = DipolSCPicard( iterative.ff.system.pos, iterative.ff.system.charges, poltens_i,
                                            iterative.ff.system.natom)
        #Set the new dipoles
        iterative.ff.part_pair_eidip.pair_pot.dipoles = newdipoles


def DipolSCPicard(pos, charges, poltens_i, natom, init=None, conv_crit=1e-10):
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
            Tensor that gives the inverse of atomic polarizabilities (3natom x 3 )
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
    G_0, G_1, G_2, D = get_ei_tensors( pos, poltens_i, natom)
    #d contains the current guess for the dipoles in following form:
    # [d0x,d0y,d0z,d1x,d1y,d1z,...]
    if init is not None:
        d = init #Set or compute initial guess
    else:
        d = np.zeros( (3*natom,) )

    converged = False
    steps = 0
    if log.do_high:
        log.hline()
        log('Starting Picard algorithm to determine dipoles')
        log('   Step     RMSD')
        log.hline()
    #Take Picard steps until convergence is achieved
    while not converged:
        d_new = - np.dot( D , ( np.dot(G_1,charges) + np.dot(G_2,d)     ) )
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

def get_ei_tensors( pos, poltens_i, natom ):
    """
    Compute tensors that help in evaluating electrostatic energy of charges and
    dipoles.
    """
    #Construct tensors, notation from Wang-Skeel 2005
    #G_0,  a [natom x natom] matrix describing the interaction between point charges and point charges
    G_0 = np.zeros( (natom,natom) )
    #Loop over first charge
    for i in xrange(natom):
        #Loop over second charge
        for j in xrange(natom):
            if j==i: continue #Exclude self-interaction
            delta = pos[i,:] - pos[j,:]     #Interatomic distance vector
            r = np.linalg.norm( delta )  #Interatomic distance
            G_0[i,j] = 1.0/r

    #G_1, a [3*natom x natom] matrix describing the interaction between point charges and point dipoles
    G_1 = np.zeros( (3*natom,natom) )
    #Loop over dipoles
    for i in xrange(natom):
        #Loop over charges
        for j in xrange(natom):
            if j==i: continue #Exclude self-interaction
            delta = pos[i,:] - pos[j,:]     #Interatomic distance vector
            r = np.linalg.norm( delta )  #Interatomic distance
            r3 = 1.0/r**3
            #Loop over x,y and z component of dipole
            for k in xrange(3):
                G_1[3*i+k,j] = delta[k]*r3

    #G_2, a [3*natom x 3*natom] matrix describing the interaction between point dipoles and point dipoles
    G_2 = np.zeros( (3*natom,3*natom) )
    #First loop over dipoles
    for i in xrange(natom):
        #Second loop over dipoles
        for j in xrange(natom):
            if j==i: continue #Exclude self-interaction
            delta = pos[i,:] - pos[j,:]     #Interatomic distance vector
            r = np.linalg.norm( delta )  #Interatomic distance
            r3 = 1.0/r**3
            r5 = 1.0/r**5
            #Loop over x,y and z component of first dipole
            for k in xrange(3):
                #Loop over x,y and z component of second dipole
                for l in xrange(3):
                    G_2[3*i+k,3*j+l] = -3.0*delta[k]*delta[l]*r5
                    if k==l: G_2[3*i+k,3*j+l] += r3

    #D, a [3*natom x 3*natom] matrix describing the atomic polarizabilities.
    #This will be a block diagonal matrix if polarizabilities are not coupled.
    #We still need to invert the given poltens_i
    D = np.linalg.inv(poltens_i)
    return G_0, G_1, G_2, D
