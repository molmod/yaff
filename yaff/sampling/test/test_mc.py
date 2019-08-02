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

def setup_gcmc_lj(L,T,fugacity):
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
        nlist = NeighborList(system, n_frame=n_frame)
        scalings = Scalings(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff = ForceField(system, [part_pair], nlist=nlist)
        return ff
    gcmc = GCMC(system, ff_generator, T, fugacity)
    return gcmc


def test_gcmc_ff_generation():
    gcmc = setup_gcmc_lj(20.0*angstrom,100,0.1)
    assert len(gcmc._ffs)==10
    for iff in range(len(gcmc._ffs)):
        assert gcmc._ffs[iff].system.natom==iff


def test_gcmc_lj():
    np.random.seed(5)
    L = 20.0*angstrom

    eos = PREOS(1.326*120.0, 0.1279*120.0*boltzmann/(3.4*angstrom)**3, 0.0)
#    eos = PREOS.from_name('argon')
    print(eos.Tc, eos.Pc/bar)
#    print(eos.Tc,1.326*120.0)
#    assert False
    sims = [(1.5*120.0,1.498*1e-3*120.0*boltzmann/(3.4*angstrom)**3,0.001*L**3/(3.4*angstrom)**3),
            (1.5*120.0,1.482*1e-2*120.0*boltzmann/(3.4*angstrom)**3,0.01*L**3/(3.4*angstrom)**3),
            (1.5*120.0,1.349*1e-1*120.0*boltzmann/(3.4*angstrom)**3,0.1*L**3/(3.4*angstrom)**3),
            ]
    for T,P,N in sims[2:]:
        fugacity = eos.calculate_fugacity(T,P)
        print("T = %8.2f P = %12.6f bar f = %12.6f bar N = %5.6f" % (T,P/bar,fugacity/bar,N))
        gcmc = setup_gcmc_lj(20.0*angstrom, T, fugacity)
        gcmc.run(50000)
        assert False
