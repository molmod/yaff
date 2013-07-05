#!/usr/bin/env python
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

# import the yaff library
from yaff import *

# Control the amount of screen output and the unit system.
log.set_level(log.medium)
log.set_unitsys(log.joule)

# import the h5py library to write output in the HDF5 format.
import h5py as h5

# 1) specify the system (32 water molecules)
system = System.from_file('system.chk')

# 2) specify the force field (experimental FF)
ff = ForceField.generate(system, 'parameters.txt')

# Open an HDF5 trajectory file for step 3 and 4. If file exists, it is overwritten.
with h5.File('output.h5', mode='w') as f:
    # 3) Integrate Newton's equation of motion and write the trajectory in HDF5
    # format.
    hdf5_writer = HDF5Writer(f)
    xyz = XYZWriter('traj.xyz')
    verlet = VerletIntegrator(ff, 1*femtosecond, hooks=[hdf5_writer, xyz], temp0=300)
    verlet.run(100)

    # 4) perform an analysis, in this case an RDF computation for O-O pairs.
    indexes = system.get_indexes('O')
    rdf = RDF(4.5*angstrom, 0.1*angstrom, f, select0=indexes)
    rdf.plot('rdf.png')
