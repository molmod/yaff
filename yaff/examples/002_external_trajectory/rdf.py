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
# --

# Needed for python2 backward compatibility
from __future__ import print_function

# Import Yaff and Numpy libraries and setup a reference system. The reference
# system is used to recognize atom types etc.
import numpy as np
from yaff import *
system = System.from_file('trajectory.xyz', rvecs=np.diag([20.3, 20.3, 20.3])*angstrom)

# Create a HDF5 file and convert the XYZ file to arrays in the HDF5 file
import h5py as h5
with  h5.File('trajectory.h5', mode='w') as f:
    system.to_hdf5(f)
    xyz_to_hdf5(f, 'trajectory.xyz')

    # Select two lists of atom indexes based on the ATSELECT rules '1' and '8'
    select0 = system.get_indexes('1')
    select1 = system.get_indexes('8')

    # Note. The remainder of the example may be moved to a separate script if
    # that would be more convenient, e.g. in case different RDFs must be generated.
    # This would avoid repetetive conversion of the XYZ file.

    # Create the RDF.
    rdf = RDF(10*angstrom, 0.1*angstrom, f, max_sample=100, select0=select0, select1=select1)
    # One may make plots with the rdf object ...
    rdf.plot()
    # ... or access the results as Numpy arrays
    print()
    print('RDF DATA FOR THE X-AXIS [A]')
    print(rdf.d/angstrom)
    print()
    print('RDF DATA FOR THE Y-AXIS')
    print(rdf.rdf)
