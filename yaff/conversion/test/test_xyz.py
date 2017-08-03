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


import h5py as h5
import pkg_resources

from yaff import *
from molmod import femtosecond


def test_xyz_to_hdf5():
    with h5.File('yaff.conversion.test.test_xyz.test_xyz_to_hdf5.h5', driver='core', backing_store=False) as f:
        # Bad practice. Proper use is to initialize the system object from a
        # different XYZ (or yet something else) with a single geometry.
        fn_xyz = pkg_resources.resource_filename(__name__, '../../data/test/water_trajectory.xyz')
        system = System.from_file(fn_xyz)
        system.to_hdf5(f)
        # Actual trajectory conversion, twice
        for i in xrange(2):
            offset = 5*i
            xyz_to_hdf5(f, fn_xyz)
            assert 'trajectory' in f
            print get_last_trajectory_row(f['trajectory'])
            for key, ds in f['trajectory'].iteritems():
                print key, ds.shape
            assert get_last_trajectory_row(f['trajectory']) == 5 + offset
            assert abs(f['trajectory/pos'][offset,0,0] - 3.340669*angstrom) < 1e-5
            assert abs(f['trajectory/pos'][-1,-1,-1] - -3.335574*angstrom) < 1e-5
            assert abs(f['trajectory/pos'][offset+3,2,1] - 3.363249*angstrom) < 1e-5


def test_xyz_to_hdf5_alt():
    with h5.File('yaff.conversion.test.test_xyz.test_xyz_to_hdf5_alt.h5', driver='core', backing_store=False) as f:
        # Bad practice. Proper use is to initialize the system object from a
        # different XYZ (or yet something else) with a single geometry.
        fn_xyz = pkg_resources.resource_filename(__name__, '../../data/test/water_trajectory.xyz')
        system = System.from_file(fn_xyz)
        system.to_hdf5(f)
        # Actual trajectory conversion, twice
        for i in xrange(2):
            offset = 5*i
            xyz_to_hdf5(f, fn_xyz, file_unit=1, name='test')
            assert 'trajectory' in f
            assert get_last_trajectory_row(f['trajectory']) == 5 + offset
            assert abs(f['trajectory/test'][offset,0,0] - 3.340669) < 1e-5
            assert abs(f['trajectory/test'][-1,-1,-1] - -3.335574) < 1e-5
            assert abs(f['trajectory/test'][offset+3,2,1] - 3.363249) < 1e-5
