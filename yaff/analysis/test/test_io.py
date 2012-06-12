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
from molmod import femtosecond


def test_xyz_to_hdf5():
    f = h5py.File('test_xyz_to_hdf5.h5', driver='core', backing_store=False)
    # Bad practice. Proper use is to initialize the system object from a
    # different XYZ (or yet something else) with a single geometry.
    system = System.from_file('input/water_trajectory.xyz')
    system.to_hdf5(f)
    # Actual trajectory conversion, twice
    for i in xrange(2):
        xyz_to_hdf5(f, 'input/water_trajectory.xyz')
        assert 'trajectory' in f
        assert f['trajectory'].attrs['row'] == 5
        assert abs(f['trajectory/pos'][0,0,0] - 3.340669*angstrom) < 1e-5
        assert abs(f['trajectory/pos'][-1,-1,-1] - -3.335574*angstrom) < 1e-5
        assert abs(f['trajectory/pos'][3,2,1] - 3.363249*angstrom) < 1e-5
    f.close()


def test_cp2k_ener_to_hdf5():
    f = h5py.File('test_xyz_to_hdf5.h5', driver='core', backing_store=False)
    # Bad practice. The trajectory file has no system directory...
    # Actual trajectory conversion, twice
    for i in xrange(2):
        cp2k_ener_to_hdf5(f, 'input/cp2k-1.ener')
        assert 'trajectory' in f
        assert f['trajectory'].attrs['row'] == 9
        assert 'step' in f['trajectory']
        assert 'time' in f['trajectory']
        assert 'ekin' in f['trajectory']
        assert 'temp' in f['trajectory']
        assert 'epot' in f['trajectory']
        assert 'econs' in f['trajectory']
        assert f['trajectory/step'][1] == 1.0
        assert abs(f['trajectory/time'][5] - 5.0*femtosecond) < 1e-10
        assert abs(f['trajectory/ekin'][-1] - 1.069191015) < 1e-5
        assert abs(f['trajectory/temp'][2] - 303.049958848) < 1e-5
        assert abs(f['trajectory/epot'][6] - -6.517529834) < 1e-5
        assert abs(f['trajectory/econs'][7] - -5.405095660) < 1e-5
    f.close()
