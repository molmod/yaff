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

from yaff import *

from yaff.pes.test.test_pair_pot import get_part_water_eidip
from yaff.sampling.polarization import *


def test_DipolSCPicard():
    #This is not really a test yet, just check if everything runs
    system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip(scalings=[0.0,1.0,1.0])
    ff = ForceField(system, [part_pair], nlist)
    opt = CGOptimizer(CartesianDOF(ff), hooks=RelaxDipoles())
    opt.run(2)

def test_polarization_get_ei_tensors():
    """Check if the tensors from polarization module give correct energy"""
    #Don't scale interactions, this is not implemented in determining the tensors
    system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip(scalings=[1.0,1.0,1.0])
    poltens_i = np.diag([1.0]*3*system.natom) #This is not used for this test
    #Get tensors from polarization module
    G_0, G_1, G_2, D = get_ei_tensors( system.pos, poltens_i, system.natom)
    #Reshape the dipole matrix to simplify matrix expressions
    dipoles = np.reshape( pair_pot.dipoles , (-1,) )
    #Compute energy using these tensors
    #Charge-charge interaction
    energy_tensor = 0.5*np.dot(np.transpose(system.charges), np.dot(G_0,system.charges) )
    #Charge-dipole interaction
    energy_tensor += np.dot( np.transpose( dipoles), np.dot( G_1, system.charges) )
    #Dipole-dipole interaction
    energy_tensor += 0.5*np.dot( np.transpose( dipoles), np.dot( G_2, dipoles) )
    #Dipole creation energy
    energy_tensor += 0.5*np.dot( np.transpose( dipoles), np.dot( np.linalg.inv(D), dipoles) )
    nlist.update() # update the neighborlists, once the rcuts are known.
    # Compute the energy using yaff.
    energy_yaff = part_pair.compute()
    assert np.abs(energy_yaff-energy_tensor) < 1.0e-10
