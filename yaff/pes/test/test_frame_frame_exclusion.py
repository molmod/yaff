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
from yaff import System, ForceField
import numpy as np
import pkg_resources

def test_exclusion():

    def random_rotation(pos):
        com = np.average(pos, axis=0)
        pos -= com
        while True:
            V1 = np.random.rand(); V2 = np.random.rand(); S = V1**2 + V2**2;
            if S < 1:
                break;
        theta = np.array([2*np.pi*(2*V1*np.sqrt(1-S)-0.5), 2*np.pi*(2*V2*np.sqrt(1-S)-0.5), np.pi*((1-2*S)/2)])
        R_x = np.array([[1, 0, 0],[0, np.cos(theta[0]), -np.sin(theta[0])],[0, np.sin(theta[0]), np.cos(theta[0])]])
        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],[0, 1, 0],[-np.sin(theta[1]), 0, np.cos(theta[1])]])
        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]),0],[np.sin(theta[2]), np.cos(theta[2]),0],[0, 0, 1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        pos_new = np.zeros((len(pos), len(pos[0])))
        for i, p in enumerate(pos):
            pos_new[i] = np.dot(R, np.array(p).T)
        return pos_new + com

    def get_adsorbate_pos(adsorbate, rvecs):
        pos = adsorbate.pos
        pos = random_rotation(pos)
        pos -= np.average(pos, axis=0)
        new_com = np.random.rand()*rvecs[0] + np.random.rand()*rvecs[1] + np.random.rand()*rvecs[2]
        return pos + new_com

    # Empty framework
    system = System.from_file(pkg_resources.resource_filename(__name__, '../../data/test/CAU_13.chk'))
    N_system = len(system.pos)
    ff_file = pkg_resources.resource_filename(__name__, '../../data/test/parameters_CAU-13_xylene.txt')

    ff = ForceField.generate(system, ff_file)
    ff.nlist.update()
    E_parts = {part.name:part.compute() for part in ff.parts}

    ff_new = ForceField.generate(system, ff_file, exclude_frame=True, n_frame=N_system)
    ff_new.nlist.update()
    E_parts_new = {part.name:part.compute() for part in ff_new.parts}

    # Add 4 adsorbates
    adsorbate = System.from_file(pkg_resources.resource_filename(__name__, '../../data/test/xylene.chk'))

    pos = system.pos
    ffatypes = np.append(system.ffatypes, adsorbate.ffatypes)
    bonds = system.bonds
    numbers = system.numbers
    ffatype_ids = system.ffatype_ids
    charges = system.charges
    masses = system.masses

    for i in range(4):
        pos = np.append(pos, get_adsorbate_pos(adsorbate,system.cell.rvecs), axis=0)
        bonds = np.append(bonds, adsorbate.bonds + N_system + len(adsorbate.pos) * i,axis=0)
        numbers = np.append(numbers, adsorbate.numbers, axis=0)
        ffatype_ids = np.append(ffatype_ids, adsorbate.ffatype_ids + max(system.ffatype_ids) + 1, axis=0)
        charges = np.append(charges, adsorbate.charges, axis=0)
        masses = np.append(masses, adsorbate.masses, axis=0)

    # Framework with 4 adsorbates
    system = System(numbers, pos, ffatypes=ffatypes, ffatype_ids=ffatype_ids, bonds=bonds,\
                    rvecs = system.cell.rvecs, charges=charges, masses=masses)

    ff = ForceField.generate(system, ff_file)
    ff_new = ForceField.generate(system, ff_file, exclude_frame=True, n_frame=N_system)

    # Test 100 random configurations
    for i in range(100):
        new_pos = ff.system.pos
        for i in range(4):
            new_pos[N_system+i*len(adsorbate.pos):N_system+(i+1)*len(adsorbate.pos)] = get_adsorbate_pos(adsorbate,system.cell.rvecs)

        ff.update_pos(new_pos)
        ff_new.update_pos(new_pos)
        ff.nlist.update()
        ff_new.nlist.update()

        E_parts_rand = {part.name:part.compute() for part in ff.parts}
        E_parts_new_rand = {part.name:part.compute() for part in ff_new.parts}
        for key, _ in E_parts.items():
            assert (E_parts[key]-E_parts_rand[key]) - (E_parts_new[key]-E_parts_new_rand[key]) < 10e-12
