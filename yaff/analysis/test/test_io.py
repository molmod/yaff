# YAFF is yet another force-field code
# Copyright (C) 2008 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
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


import h5py

from yaff import *


def test_xyz_to_hdf5():
    f = h5py.File('tmp.h5', driver='core', backing_store=False)
    # Bad practice. Proper use is to initialize the system object from a
    # different XYZ (or yet something else) with a single geometry.
    system = System.from_file('input/water_trajectory.xyz')
    system.to_hdf5(f)
    # Actual trajectory conversion
    xyz_to_hdf5(f, 'input/water_trajectory.xyz')
    assert 'trajectory' in f
    assert f['trajectory'].attrs['row'] == 5
    assert abs(f['trajectory/pos'][0,0,0] - 3.340669*angstrom) < 1e-5
    assert abs(f['trajectory/pos'][-1,-1,-1] - -3.335574*angstrom) < 1e-5
    assert abs(f['trajectory/pos'][3,2,1] - 3.363249*angstrom) < 1e-5
    f.close()
