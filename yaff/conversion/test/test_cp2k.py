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


from __future__ import division

import h5py as h5
import pkg_resources

from yaff import *
from molmod import femtosecond


def test_cp2k_ener_to_hdf5():
    with h5.File(__name__ + '.test_xyz_to_hdf5.h5', driver='core', backing_store=False) as f:
        # Bad practice. The trajectory file has no system directory...
        # Actual trajectory conversion, twice
        for i in range(2):
            offset = i*9
            fn = pkg_resources.resource_filename(__name__, '../../data/test/cp2k-1.ener')
            cp2k_ener_to_hdf5(f, fn)
            assert 'trajectory' in f
            assert get_last_trajectory_row(f['trajectory']) == 9 + offset
            assert 'step' in f['trajectory']
            assert 'time' in f['trajectory']
            assert 'ekin' in f['trajectory']
            assert 'temp' in f['trajectory']
            assert 'epot' in f['trajectory']
            assert 'econs' in f['trajectory']
            assert f['trajectory/step'][1] == 1.0
            assert abs(f['trajectory/time'][5+offset] - 5.0*femtosecond) < 1e-10
            assert abs(f['trajectory/ekin'][-1] - 1.069191015) < 1e-5
            assert abs(f['trajectory/temp'][2+offset] - 303.049958848) < 1e-5
            assert abs(f['trajectory/epot'][6+offset] - -6.517529834) < 1e-5
            assert abs(f['trajectory/econs'][7+offset] - -5.405095660) < 1e-5
