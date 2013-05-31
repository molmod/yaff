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

from yaff import *


def test_dlpoly_history_uo():
    f = h5.File('yaff.conversion.test.test_dlpoly.test_dlpoly_history_uo.h5', driver='core', backing_store=False)
    # Bad practice. The trajectory file has no system directory...
    # Actual trajectory conversion, twice
    for i in xrange(2):
        fn = context.get_fn('test/dlpoly_HISTORY_uo')
        dlpoly_history_to_hdf5(f, fn)
        assert 'trajectory' in f
        assert f['trajectory'].attrs['row'] == 3
        assert abs(f['trajectory/time'][0]/picosecond - 4.00) < 1e-10
        assert abs(f['trajectory/time'][1]/picosecond - 4.05) < 1e-10
        assert f['trajectory/pos'].shape == (3, 324, 3)
        assert f['trajectory/vel'].shape == (3, 324, 3)
        assert f['trajectory/frc'].shape == (3, 324, 3)

        assert abs(f['trajectory/cell'][0,0,0]/angstrom - 16.46) < 1e-10
        assert abs(f['trajectory/pos'][0,0]/angstrom - [1.3522E+00, 1.3159E+00, 1.4312E+00]).max() < 1e-10
        assert abs(f['trajectory/vel'][0,0]/angstrom*picosecond - [1.5113E+01, 1.0559E+00, 1.2843E-01]).max() < 1e-10
        assert abs(f['trajectory/frc'][0,0]/(amu*angstrom/picosecond**2) - [1.7612E+03, 3.6680E+03, 2.4235E+03]).max() < 1e-10

        assert abs(f['trajectory/cell'][1,2,2]/angstrom - 16.46) < 1e-10
        assert abs(f['trajectory/pos'][1,2]/angstrom - [1.2991E+00, 1.2788E+00, -4.1091E+00]).max() < 1e-10
        assert abs(f['trajectory/vel'][1,3]/angstrom*picosecond - [-1.1137E+01, 7.0935E+00, 1.0432E+01]).max() < 1e-10
        assert abs(f['trajectory/frc'][1,3]/(amu*angstrom/picosecond**2) - [-4.5523E+03, -2.4187E+03, -3.1966E+03]).max() < 1e-10


def test_dlpoly_history_sam():
    f = h5.File('yaff.conversion.test.test_dlpoly.test_dlpoly_history_sam.h5', driver='core', backing_store=False)
    # Bad practice. The trajectory file has no system directory...
    # Actual trajectory conversion, twice
    for i in xrange(2):
        fn = context.get_fn('test/dlpoly_HISTORY_sam')
        dlpoly_history_to_hdf5(f, fn)
        assert 'trajectory' in f
        assert f['trajectory'].attrs['row'] == 3
        assert abs(f['trajectory/time'][0]/picosecond - 0.500) < 1e-10
        assert abs(f['trajectory/time'][1]/picosecond - 0.501) < 1e-10
        assert f['trajectory/pos'].shape == (3, 1264, 3)
        assert 'trajectory/vel' not in f
        assert 'trajectory/frc' not in f

        assert abs(f['trajectory/cell'][0]/angstrom - [[35.47, 0.0, 0.0], [17.69, 30.7, 0.0], [17.7, 10.33, 28.83]]).max() < 1e-10
        assert abs(f['trajectory/pos'][0,0]/angstrom - [1.1370E+01, 1.3308E+01, 4.6682E-01]).max() < 1e-10

        assert abs(f['trajectory/cell'][1,2,2]/angstrom - 28.83) < 1e-10
        assert abs(f['trajectory/pos'][1,2]/angstrom - [1.4848E+01, 5.1697E+00, 1.2132E+01]).max() < 1e-10
