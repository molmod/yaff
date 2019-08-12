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

from yaff import *
from molmod.units import angstrom, bar, kelvin, kcalmol
from molmod.constants import boltzmann

def setup_gcmc_lj(L):
    '''GCMC for a cubic box of length L, with single-site Lennard-Jones particles'''
    numbers = np.ones((1,), dtype=int)*18
    pos = np.zeros((1,3))
    rvecs = np.eye(3)*L
    system = System(numbers, pos, rvecs=rvecs, bonds=np.zeros((0,2)))
    def ff_generator(system, guest):
        natom = system.natom
        sigma = 3.4*angstrom
        epsilons = np.ones((system.natom,))*120.0*boltzmann
        sigmas = np.ones((system.natom,))*sigma
        rcut = 2.5*sigma
        pair_pot = PairPotLJ(sigmas, epsilons, rcut, None)
        n_frame = system.natom-guest.natom
        if system.natom==0: n_frame = 0
        nlist = NeighborList(system, nlow=n_frame, nhigh=n_frame)
        scalings = Scalings(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff = ForceField(system, [part_pair], nlist=nlist)
        return ff
    gcmc = GCMC(system, ff_generator)
    return gcmc


def test_gcmc_ff_generation():
    gcmc = setup_gcmc_lj(20.0*angstrom)
    assert len(gcmc._ffs)==10
    for iff in range(len(gcmc._ffs)):
        assert gcmc._ffs[iff].system.natom==iff


def test_gcmc_lj():
    L = 20.0*angstrom
    eos = PREOS(1.326*120.0, 0.1279*120.0*boltzmann/(3.4*angstrom)**3, 0.0)
    eos = PREOS.from_name('argon')
    # These are simulation data taken from
    # https://www.nist.gov/mml/csd/informatics/lammps-md-equation-state-pressure-vs-density-linear-force-shifted-potential-25s
    sims = [(1.5*120.0,1.498*1e-3*120.0*boltzmann/(3.4*angstrom)**3,0.001*L**3/(3.4*angstrom)**3),
            (1.5*120.0,1.482*1e-2*120.0*boltzmann/(3.4*angstrom)**3,0.01*L**3/(3.4*angstrom)**3),
            (1.5*120.0,1.349*1e-1*120.0*boltzmann/(3.4*angstrom)**3,0.1*L**3/(3.4*angstrom)**3),
            ]
    gcmc = setup_gcmc_lj(20.0*angstrom)
    for T,P,N in sims[:]:
        fugacity = eos.calculate_fugacity(T,P)
        gcmc.set_external_conditions(T, fugacity)
        gcmc.run(10000)
        relerr = gcmc.Nmean/N - 1.0
        # Don't expect well converged results for such short simulations
        # This only serves to show that the results are not ridiculous, rather
        # than demonstrating that the phase space is correctly sampled
        print("N(ref) = %8.4f N(sim) = %8.4f Relative error = %8.2f %%" %
            (N, gcmc.Nmean, relerr*100.0) )
        assert np.abs(relerr)<0.2


def test_gcmc_probabilities():
    gcmc = setup_gcmc_lj(20.0*angstrom)
    # We're not interested in the simulation results, so we choose a low
    # pressure to get a low number of guests and faster simulation.
    gcmc.set_external_conditions(200.0,0.01*bar)
    mc_moves = {'insertion':0.35, 'deletion':0.15,
                      'translation': 0.55, 'rotation':0.2}
    nsteps = 9800
    acceptance = gcmc.run(nsteps, mc_moves = mc_moves)
    # Normalize the provided probabilities
    ptotal = sum([p for t,p in mc_moves.items()])
    for i, t in enumerate(sorted(mc_moves.keys())):
        p = mc_moves[t]
#        print("Move: %20s P(ref) = %8.2f %% P(sim) = %8.2f %%" %
#         (t, p/ptotal*100, float(acceptance[i,1])/nsteps*100) )
        assert np.abs(p/ptotal-float(acceptance[i,1])/nsteps)<5e-2
