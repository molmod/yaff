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

from molmod import kcalmol, angstrom, rad, deg
from yaff import *

from common import get_system_water32, check_gradient_ff, check_gradient_part

def get_ff_water32(do_valence=False, do_lj=False, do_eireal=False, do_eireci=False):
    system = get_system_water32()
    cutoff = 9*angstrom
    alpha = 4.5/cutoff
    scalings = Scalings(system.topology)
    parts = []
    if do_valence:
        # Valence part
        vpart = ValencePart(system)
        for i, j in system.topology.bonds:
            vpart.add_term(Harmonic(450.0*kcalmol/angstrom**2, 0.9572*angstrom, Bond(i, j)))
        for i1 in xrange(system.natom):
            for i0 in system.topology.neighs1[i1]:
                for i2 in system.topology.neighs1[i1]:
                    if i0 > i2:
                        vpart.add_term(Harmonic(55.000*kcalmol/rad**2, 104.52*deg, BendAngle(i0, i1, i2)))
        parts.append(vpart)
    if do_lj or do_eireal:
        # Neighbor lists, scalings
        nlists = NeighborLists(system)
    else:
        nlists = None
    if do_lj:
        # Lennard-Jones part
        rminhalf_table = {1: 0.2245*angstrom, 8: 1.7682*angstrom}
        epsilon_table = {1: -0.0460*kcalmol, 8: -0.1521*kcalmol}
        sigmas = np.zeros(96, float)
        epsilons = np.zeros(96, float)
        for i in xrange(system.natom):
            sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
            epsilons[i] = epsilon_table[system.numbers[i]]
        pair_pot_lj = PairPotLJ(sigmas, epsilons, cutoff, True)
        pair_part_lj = PairPart(nlists, scalings, pair_pot_lj)
        #check_gradient_part(system, pair_part_lj, 1e-100, nlists)
        #raise Exception
        parts.append(pair_part_lj)
    # charges
    q0 = 0.417
    charges = -2*q0 + (system.numbers == 1)*3*q0
    assert abs(charges.sum()) < 1e-8
    if do_eireal:
        # Real-space electrostatics
        pair_pot_ei = PairPotEI(charges, alpha, cutoff)
        pair_part_ei = PairPart(nlists, scalings, pair_pot_ei)
        parts.append(pair_part_ei)
    if do_eireci:
        # Reciprocal-space electrostatics
        gmax = np.ceil(alpha*1.5/system.gspacings-0.5).astype(int)
        ewald_reci_part = EwaldReciprocalPart(system, charges, alpha, gmax)
        parts.append(ewald_reci_part)
        # Ewald corrections
        ewald_corr_part = EwaldCorrectionPart(system, charges, alpha, scalings)
        parts.append(ewald_corr_part)
    return SumForceField(system, parts, nlists)


def test_gradient_water32_full():
    ff = get_ff_water32(True, True, True, True)
    check_gradient_ff(ff, 1e-10)
