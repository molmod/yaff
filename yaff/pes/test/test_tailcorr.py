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

import numpy as np
from scipy.integrate import quadrature

from molmod import kcalmol, angstrom, rad, deg, femtosecond, boltzmann, pascal
mpa = 1e6*pascal
from molmod.periodic import periodic

from yaff import *
from yaff.test.common import get_system_fluidum_grid, get_system_water32




def check_tailcorr_convergence(system, pairpot_class, decay, n_frame, *args, **kwargs):
    """
    Systematically increase rcut to extrapolate the energy and pressure for
    rcut -> infinity. Then check that the energy and pressure including
    tailcorrections are close to this limiting value for all rcuts.

    Arguments
        * system: a YAFF System instance
        * pairpot_class: a PairPot class (NOT an instance of it)
        * decay: the long range behavior of the energy, should be
            ``r6'' for potentials decaying as r**(-6)
            or ``exp'' for exponentially decaying potentials
        * *args: all arguments that need to be passed to initialize the
          pairpot_class, except for rcut
        * **kwargs: all keyword arguments that need to be passed to initialize
          the pairpot_class
    """
    nlist = NeighborList(system, n_frame=n_frame)
    scalings = Scalings(system)
    rcuts = np.linspace(12.0,32.0,11)*angstrom
    vtens0, vtens1 = np.zeros((3,3)), np.zeros((3,3))
    data = np.zeros((rcuts.shape[0],4))
    # Loop over all values of rcut, collect energies, pressures and their
    # tailcorrections in the data array
    for ircut, rcut in enumerate(rcuts):
        newargs = args + (rcut,)
        pair_pot = pairpot_class(*(newargs), **kwargs)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        part_tailcorr = ForcePartTailCorrection(system, part_pair, n_frame=n_frame)
        ff0 = ForceField(system, [part_pair], nlist=nlist)
        ff1 = ForceField(system, [part_pair, part_tailcorr], nlist=nlist)
        vtens0[:] = 0.0
        e0 = ff0.compute(vtens=vtens0)
        p0 = np.trace(vtens0)/3.0/ff0.system.cell.volume
        vtens1[:] = 0.0
        e1 = ff1.compute(vtens=vtens1)
        p1 = np.trace(vtens1)/3.0/ff1.system.cell.volume
        row = [e0,e1,p0,p1]
        data[ircut] = row
        print("%20.12f | %20.12f %20.12f | %20.12f %20.12f"%(rcut/angstrom,e0/kcalmol,(e1)/kcalmol,p0/mpa,(p1)/mpa))
    # Find the reference energy and pressure
    if decay=='exp':
        # If the potential decays exponentially, the energy and pressure will
        # be approximately constant as a function of rcut (for reasonable
        # values of rcut)
        eref, pref = data[-1,0], data[-1,2]
    elif decay=='r6':
        # If the potential decays as r**-6, the energy and pressure will show
        # a r**-3 dependence on rcut
        ecoeffs = np.polyfit(1.0/rcuts**3, data[:,0], 1)
        pcoeffs = np.polyfit(1.0/rcuts**3, data[:,2], 1)
        eref = ecoeffs[1]
        pref = pcoeffs[1]
    else: raise NotImplementedError
    print("eref = %20.12f pref = %20.12f"%(eref/kcalmol,pref/mpa))
    print("edev = %20.12f pdev = %20.12f"%(np.amax(np.abs(data[:,1]-eref))/kcalmol,np.amax(np.abs(data[:,3]-pref))/mpa))
    print("="*100)
    assert np.amax(np.abs(data[:,1]-eref)) < 0.05*kcalmol
    assert np.amax(np.abs(data[:,3]-pref)) < 0.1*mpa


def check_tailcorr_numerical_singlepair(pairpot_class, rcut, rmin, *args, **kwargs):
    '''
    Check the implementation of tailcorrections for a single pair of identical
    atoms by comparing with numerical integration.
    '''
    # Construct a system with a single pair of atoms
    numbers = np.array([1,1], dtype=int)
    pos = np.zeros((numbers.shape[0],3))
    ffatypes = ['A','A']
    system = System(numbers, pos, ffatypes=ffatypes, bonds=[])
    nlist = NeighborList(system)
    scalings = Scalings(system)
    # Becke integration grid (https://pamoc.eu/tpc_num_int.html#GLQ)
    R = 5.0*angstrom # Controls the extent of the grid, but is not equal to the largest value
    npoints = 100
    i = np.arange(npoints)
    q = np.cos( (i+1)*np.pi/(npoints+1))
    rgrid = rmin + R*(1+q)/(1-q)
    weights = 2.0*np.pi/(npoints+1)*(rgrid-rmin)/np.sqrt(1.0-q**2)
    # Set up a pairpot with truncation
    newargs = args + (rcut,)
    pairpot_tr = pairpot_class(*(newargs), **kwargs)
    ecorr, wcorr = pairpot_tr.prepare_tailcorrections(system.natom)
    part_tr = ForcePartPair(system, nlist, scalings, pairpot_tr)
    # Set up a pairpot with truncation larger than the largest considered r value
    newargs = args + (2*np.amax(rgrid),)
    pairpot = pairpot_class(*(newargs), **kwargs)
    part = ForcePartPair(system, nlist, scalings, pairpot)
    # Do the numerical integration
    nlist.update()
    gpos0, gpos1 = np.zeros((system.natom,3)), np.zeros((system.natom,3))
    data = np.zeros((rgrid.shape[0],2))
    for ir, r in enumerate(rgrid):
        nlist.neighs[0]['d'] = r
        nlist.neighs[0]['dz'] = r
        gpos0[:] = 0.0
        gpos1[:] = 0.0
        e0,e1 = part_tr.compute(gpos0), part.compute(gpos1)
        row = [(e1-e0)*r**2,(gpos1[0,2]-gpos0[0,2])*r**3/3.0]
        data[ir] = row
    ecorr_num = 4.0*np.sum(weights*data[:,0])
    wcorr_num = 4.0*np.sum(weights*data[:,1])
    print("%20.12f %20.12f %20.12f %20.12f" % (ecorr, ecorr_num, ecorr-ecorr_num, ecorr_num/ecorr-1.0))
    print("%20.12f %20.12f %20.12f %20.12f" % (wcorr, wcorr_num, wcorr-wcorr_num, wcorr_num/wcorr-1.0))
    # Check absolute error
    assert np.abs(ecorr-ecorr_num)<1e-5
    assert np.abs(wcorr-wcorr_num)<1e-5
    # Check relative errors
    if np.abs(ecorr)>1e-6:
        assert np.abs(ecorr_num/ecorr-1.0) < 1e-3
        assert np.abs(wcorr_num/wcorr-1.0) < 1e-3

def test_tailcorr_numericalint_lj():
    sigmas = np.array([2.1,2.1])*angstrom
    epsilons = np.array([0.1,0.1])*kcalmol
    rcut = 12.0*angstrom
    width = 3.0*angstrom
    for tr,rmin in zip([None, Switch3(width)],[rcut,rcut-width]):
        check_tailcorr_numerical_singlepair(PairPotLJ, rcut, rmin, sigmas, epsilons, tr=tr)


def test_tailcorr_numericalint_mm3():
    sigmas = np.array([2.1,2.1])*angstrom
    epsilons = np.array([0.1,0.1])*kcalmol
    rcut = 12.0*angstrom
    width = 3.0*angstrom
    for onlypauli in [0,1]:
        onlypaulis = np.ones((sigmas.shape[0],),dtype=np.int32)*onlypauli
        for tr,rmin in zip([None, Switch3(width)],[rcut,rcut-width]):
            check_tailcorr_numerical_singlepair(PairPotMM3, rcut, rmin, sigmas, epsilons, onlypaulis, tr=tr)


def test_tailcorr_fluidum_lj():
    natom = 64
    system = get_system_fluidum_grid(natom)
    nlist = NeighborList(system)
    scalings = Scalings(system)
    sigmas = np.ones((natom,))*2.1*angstrom
    sigmas += np.random.normal(0.0,0.01*angstrom,sigmas.shape[0])
    epsilons = np.ones((natom,))*0.15*kcalmol
    epsilons += np.random.normal(0.0,0.001*kcalmol,epsilons.shape[0])
    for tr in [None,Switch3(3.0*angstrom)]:
        check_tailcorr_convergence(system, PairPotLJ, 'r6', 0, sigmas, epsilons, tr=tr)


def test_tailcorr_fluidum_lj_exclude_frame():
    natom = 64
    system = get_system_fluidum_grid(natom)
    nlist = NeighborList(system, n_frame=59)
    scalings = Scalings(system)
    sigmas = np.ones((natom,))*2.1*angstrom
    sigmas += np.random.normal(0.0,0.01*angstrom,sigmas.shape[0])
    epsilons = np.ones((natom,))*0.15*kcalmol
    epsilons += np.random.normal(0.0,0.001*kcalmol,epsilons.shape[0])
    for tr in [None,Switch3(3.0*angstrom)]:
        check_tailcorr_convergence(system, PairPotLJ, 'r6', 59, sigmas, epsilons, tr=tr)


def test_tailcorr_fluidum_mm3():
    natom = 64
    system = get_system_fluidum_grid(natom)
    nlist = NeighborList(system)
    scalings = Scalings(system)
    sigmas = np.ones((natom,))*1.1*angstrom
    sigmas += np.random.normal(0.0,0.01*angstrom,sigmas.shape[0])
    epsilons = np.ones((natom,))*0.15*kcalmol
    epsilons += np.random.normal(0.0,0.001*kcalmol,epsilons.shape[0])
    for onlypauli,decay in zip([0,1],['r6','exp']):
        onlypaulis = np.ones((natom,),dtype=np.int32)*onlypauli
        for tr in [None,Switch3(3.0*angstrom)]:
            check_tailcorr_convergence(system, PairPotMM3, decay, 0, sigmas, epsilons, onlypaulis, tr=tr)


def test_tailcorr_fluidum_grimme():
    natom = 64
    system = get_system_fluidum_grid(natom)
    nlist = NeighborList(system)
    scalings = Scalings(system)
    r0s = np.ones((natom,))*1.1*angstrom
    r0s += np.random.normal(0.0,0.01*angstrom,r0s.shape[0])
    c6s = np.ones((natom,))*0.2*1e-3*kjmol*nanometer**6
    c6s += np.random.normal(0.0,0.01*angstrom,c6s.shape[0])
    for tr in [None,Switch3(3.0*angstrom)]:
        check_tailcorr_convergence(system, PairPotGrimme, 'r6', 0, r0s, c6s, tr=tr)


def test_tailcorr_fluidum_exprep():
    natom = 64
    system = get_system_fluidum_grid(natom, ffatypes=['A','B','C'])
    nlist = NeighborList(system)
    scalings = Scalings(system)
    # Initialize parameters
    amps = np.array([2.0,3.0,4.0])
    bs = np.array([4.5,4.6,4.7])/angstrom
    # Allocate some arrays for the pair potential
    assert len(system.ffatypes) == 3
    amp_cross = np.zeros((3, 3), float)
    b_cross = np.zeros((3, 3), float)
    amp_mix, amp_mix_coeff, b_mix, b_mix_coeff = 0,0,1,0
    for tr in [None,Switch3(3.0*angstrom)]:
        check_tailcorr_convergence(system, PairPotExpRep, 'exp', 0,
            system.ffatype_ids, amp_cross, b_cross, tr=tr,
            amps=amps, amp_mix=amp_mix, amp_mix_coeff=amp_mix_coeff,
            bs=bs, b_mix=b_mix, b_mix_coeff=b_mix_coeff)


def test_tailcorr_fluidum_ljcross():
    system = get_system_fluidum_grid(64, ffatypes=['A','B','C','D'])
    nlist = NeighborList(system)
    scalings = Scalings(system)
    assert system.nffatype == 4
    eps_cross = np.array([[1.0,3.5,4.6,9.4],
                          [3.5,2.0,4.4,4.1],
                          [4.6,4.4,5.0,3.3],
                          [9.4,4.1,3.3,0.1]])*kcalmol
    sig_cross = np.array([[0.2,0.5,1.6,1.4],
                          [0.5,1.0,2.4,1.1],
                          [1.6,2.4,1.0,2.3],
                          [1.4,1.1,2.3,0.9]])
    assert np.all( np.abs( sig_cross - np.transpose(sig_cross) ) < 1e-15 )
    assert np.all( np.abs( eps_cross - np.transpose(eps_cross) ) < 1e-15 )
    for tr in [None,Switch3(3.0*angstrom)]:
        check_tailcorr_convergence(system, PairPotLJCross, 'r6', 0,
            system.ffatype_ids, eps_cross, sig_cross, tr=tr)
