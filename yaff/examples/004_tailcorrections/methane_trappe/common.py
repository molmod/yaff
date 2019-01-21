#!/usr/bin/env python
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

import numpy as np
import h5py as h5

from yaff import log
from yaff.pes import NeighborList, Scalings, PairPotLJ, ForcePartPair, Switch3,\
    ForceField, ForcePart, ForcePartTailCorrection
from yaff.test.common import get_system_fluidum_grid
from molmod.periodic import periodic
from molmod.units import meter, kilogram, angstrom, kelvin
from molmod.constants import boltzmann


'''
Simulate liquid methane using the TraPPE force field, which represents methane
molecules as single sites interacting through a Lennard-Jones potential.

TraPPE parameters can be found here:
    http://chem-siepmann.oit.umn.edu/siepmann/trappe/index.html
or in the paper
    Martin and Siepmann,
    Transferable Potentials for Phase Equilibria. 1. United-Atom Description of n-Alkanes
    J. Phys. Chem. B, 1998, 102 (14), pp 2569–2577
    http://dx.doi.org/10.1021/jp972543+
'''


__all__ = ['get_ff', 'get_system']


def get_ff(system, rcut, tr=Switch3(2.0*angstrom), tailcorr=False):
    sigmas = 3.730*np.ones((system.natom,))*angstrom
    epsilons = 148.0*kelvin*boltzmann*np.ones((system.natom))
    nlist = NeighborList(system)
    scalings = Scalings(system)
    pair_lj = PairPotLJ(sigmas, epsilons, rcut, tr=tr)
    part_lj = ForcePartPair(system, nlist, scalings, pair_lj)
    parts = [part_lj]
    if tailcorr:
        part_tail = ForcePartTailCorrection(system, part_lj)
        parts.append(part_tail)
    ff = ForceField(system, parts, nlist)
    return ff


def get_system(nmol, rho0=420.0*kilogram/meter**3):
    # Initial density
    masses = np.ones((nmol,))*( periodic[6].mass + 4*periodic[1].mass )
    # Initial volume
    V0 = np.sum(masses)/rho0
    # Estimate intermolecular distance based on the volume per molecule
    l0 = (V0/nmol)**(1.0/3.0)
    # Setup the system
    system = get_system_fluidum_grid(nmol, ffatypes=['CH4'], l0=l0)
    system.masses = masses
    return system
