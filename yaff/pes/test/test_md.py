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


import numpy as np

from molmod import kcalmol, angstrom, rad, deg, femtosecond, boltzmann
from molmod.periodic import periodic

from yaff import *

from yaff.test.common import get_system_water32
from yaff.pes.test.common import check_gpos_ff, check_vtens_ff


def get_ff_water32(do_valence=False, do_lj=False, do_eireal=False, do_eireci=False):
    tr = Switch3(3*angstrom)
    system = get_system_water32()
    rcut = 7*angstrom
    alpha = 4.5/rcut
    scalings = Scalings(system)
    parts = []
    if do_valence:
        # Valence part
        part_valence = ForcePartValence(system)
        for i, j in system.bonds:
            part_valence.add_term(Harmonic(450.0*kcalmol/angstrom**2, 0.9572*angstrom, Bond(i, j)))
        for i1 in xrange(system.natom):
            for i0 in system.neighs1[i1]:
                for i2 in system.neighs1[i1]:
                    if i0 > i2:
                        part_valence.add_term(Harmonic(55.000*kcalmol/rad**2, 104.52*deg, BendAngle(i0, i1, i2)))
        parts.append(part_valence)
    if do_lj or do_eireal:
        # Neighbor lists, scalings
        nlist = NeighborList(system, skin=2*angstrom)
    else:
        nlist = None
    if do_lj:
        # Lennard-Jones part
        rminhalf_table = {1: 0.2245*angstrom, 8: 1.7682*angstrom}
        epsilon_table = {1: -0.0460*kcalmol, 8: -0.1521*kcalmol}
        sigmas = np.zeros(96, float)
        epsilons = np.zeros(96, float)
        for i in xrange(system.natom):
            sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
            epsilons[i] = epsilon_table[system.numbers[i]]
        pair_pot_lj = PairPotLJ(sigmas, epsilons, rcut, tr)
        part_pair_lj = ForcePartPair(system, nlist, scalings, pair_pot_lj)
        parts.append(part_pair_lj)
    # electrostatics
    if do_eireal:
        # Real-space electrostatics
        pair_pot_ei = PairPotEI(system.charges, alpha, rcut)
        part_pair_ei = ForcePartPair(system, nlist, scalings, pair_pot_ei)
        parts.append(part_pair_ei)
    if do_eireci:
        # Reciprocal-space electrostatics
        part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=alpha/0.75)
        parts.append(part_ewald_reci)
        # Ewald corrections
        part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings)
        parts.append(part_ewald_corr)
    return ForceField(system, parts, nlist)


def test_gpos_water32_full():
    ff = get_ff_water32(True, True, True, True)
    check_gpos_ff(ff)


def test_vtens_water32_full():
    ff = get_ff_water32(True, True, True, True)
    check_vtens_ff(ff)


def test_md_water32_full():
    ff = get_ff_water32(True, True, True, True)
    pos = ff.system.pos.copy()
    grad = np.zeros(pos.shape)
    h = 1.0*femtosecond
    mass = np.array([periodic[n].mass for n in ff.system.numbers]).reshape((-1,1))
    # init
    ff.update_pos(pos)
    epot = ff.compute(grad)
    temp = 300
    vel = np.random.normal(0, 1, pos.shape)*np.sqrt((2*boltzmann*temp)/mass)
    velh = vel + (-0.5*h)*grad/mass
    # prop
    cqs = []
    symbols = [ff.system.get_ffatype(i) for i in xrange(ff.system.natom)]
    for i in xrange(100):
        pos += velh*h
        ff.update_pos(pos)
        grad[:] = 0.0
        epot = ff.compute(grad)
        tmp = (-0.5*h)*grad/mass
        vel = velh + tmp
        ekin = 0.5*(mass*vel*vel).sum()
        cqs.append(ekin + epot)
        velh = vel + tmp
    cqs = np.array(cqs)
    assert cqs.std() < 5e-3
