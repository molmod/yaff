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


from nose.plugins.skip import SkipTest
import pkg_resources
import h5py as h5

from yaff import *


def test_dlpoly_history_uo():
    f = h5.File('yaff.conversion.test.test_dlpoly.test_dlpoly_history_uo.h5', driver='core', backing_store=False)
    # Bad practice. The trajectory file has no system directory...
    # Actual trajectory conversion, twice
    for i in xrange(2):
        offset = 3*i
        fn = pkg_resources.resource_filename(__name__, '../../data/test/dlpoly_HISTORY_uo')
        dlpoly_history_to_hdf5(f, fn)
        assert 'trajectory' in f
        assert get_last_trajectory_row(f['trajectory']) == 3 + offset
        assert abs(f['trajectory/time'][offset]/picosecond - 4.00) < 1e-10
        assert abs(f['trajectory/time'][offset+1]/picosecond - 4.05) < 1e-10
        assert f['trajectory/pos'].shape == (3+offset, 3, 3)
        assert f['trajectory/vel'].shape == (3+offset, 3, 3)
        assert f['trajectory/frc'].shape == (3+offset, 3, 3)

        assert abs(f['trajectory/cell'][offset,0,0]/angstrom - 16.46) < 1e-10
        assert abs(f['trajectory/pos'][offset,0]/angstrom - [1.3522E+00, 1.3159E+00, 1.4312E+00]).max() < 1e-10
        assert abs(f['trajectory/vel'][offset,0]/angstrom*picosecond - [1.5113E+01, 1.0559E+00, 1.2843E-01]).max() < 1e-10
        assert abs(f['trajectory/frc'][offset,0]/(amu*angstrom/picosecond**2) - [1.7612E+03, 3.6680E+03, 2.4235E+03]).max() < 1e-10

        assert abs(f['trajectory/cell'][offset+1,2,2]/angstrom - 16.46) < 1e-10
        assert abs(f['trajectory/pos'][offset+1,2]/angstrom - [-6.2693E-03, -2.4735E-02, 1.2793E-02]).max() < 1e-10
        assert abs(f['trajectory/vel'][offset+1,1]/angstrom*picosecond - [7.0023E-01, -9.6551E+00, -1.1618E+01]).max() < 1e-10
        assert abs(f['trajectory/frc'][offset+1,2]/(amu*angstrom/picosecond**2) - [7.9765E+03, 3.5419E+01, 2.6775E+03]).max() < 1e-10


def test_dlpoly_history_sam():
    f = h5.File('yaff.conversion.test.test_dlpoly.test_dlpoly_history_sam.h5', driver='core', backing_store=False)
    # Bad practice. The trajectory file has no system directory...
    # Actual trajectory conversion, twice
    for i in xrange(2):
        offset = 3*i
        fn = pkg_resources.resource_filename(__name__, '../../data/test/dlpoly_HISTORY_sam')
        dlpoly_history_to_hdf5(f, fn)
        assert 'trajectory' in f
        assert get_last_trajectory_row(f['trajectory']) == 3+offset
        assert abs(f['trajectory/time'][offset]/picosecond - 0.500) < 1e-10
        assert abs(f['trajectory/time'][offset+1]/picosecond - 0.501) < 1e-10
        assert f['trajectory/pos'].shape == (3+offset, 15, 3)
        assert 'trajectory/vel' not in f
        assert 'trajectory/frc' not in f

        assert abs(f['trajectory/cell'][offset]/angstrom - [[35.47, 0.0, 0.0], [17.69, 30.7, 0.0], [17.7, 10.33, 28.83]]).max() < 1e-10
        assert abs(f['trajectory/pos'][offset,0]/angstrom - [1.1370E+01, 1.3308E+01, 4.6682E-01]).max() < 1e-10

        assert abs(f['trajectory/cell'][offset+1,2,2]/angstrom - 28.83) < 1e-10
        assert abs(f['trajectory/pos'][offset+1,2]/angstrom - [1.4848E+01, 5.1697E+00, 1.2132E+01]).max() < 1e-10


def test_dlpoly_history_an():
    raise SkipTest('Fails, ask An why because dlpoly_history_sam does work')
    with h5.File('yaff.conversion.test.test_dlpoly.test_dlpoly_history_an.h5', driver='core', backing_store=False) as f:
        # Bad practice. The trajectory file has no system directory...
        # Actual trajectory conversion, par1
        fn = pkg_resources.resource_filename(__name__, '../../data/test/dlpoly_HISTORY_an1')
        dlpoly_history_to_hdf5(f, fn)
        assert get_last_trajectory_row(f['trajectory']) == 2
        assert abs(f['trajectory/cell'][1]/angstrom - [[27.41, 0.000, 0.000], [-13.71, 23.84, 0.000], [-0.1021E-01, 0.9013E-02, 29.50]]).max() < 1e-10
        assert abs(f['trajectory/pos'][0,-1,-1]/angstrom - -1.4007E+01) < 1e-10
        # Actual trajectory conversion, par1
        fn = pkg_resources.resource_filename(__name__, '../../data/test/dlpoly_HISTORY_an2')
        dlpoly_history_to_hdf5(f, fn)
        assert get_last_trajectory_row(f['trajectory']) == 4
        assert abs(f['trajectory/cell'][1]/angstrom - [[27.41, 0.000, 0.000], [-13.71, 23.84, 0.000], [-0.1021E-01, 0.9013E-02, 29.50]]).max() < 1e-10
        assert abs(f['trajectory/cell'][3]/angstrom - [[27.41, 0.000, 0.000], [-13.71, 23.84, 0.000], [-0.1021E-01, 0.9013E-02, 29.50]]).max() < 1e-10
        assert abs(f['trajectory/pos'][0,-1,-1]/angstrom - -1.4007E+01) < 1e-10
        assert abs(f['trajectory/pos'][3,-1,-1]/angstrom - 1.4275E+01) < 1e-10
