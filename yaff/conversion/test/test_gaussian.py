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
import h5py as h5
import pkg_resources

from yaff import *
from yaff.conversion.gaussian import _scan_g09_forces, _scan_g09_time, \
    _scan_g09_pos_vel, _scan_to_line


def test_scan_forces():
    fn_log = pkg_resources.resource_filename(__name__, '../../data/test/gaussian_sioh4_md.log')
    with open(fn_log) as f:
        numbers, frc = _scan_g09_forces(f)

    assert numbers[0] == 14
    assert numbers[1] == 8
    assert numbers[-1] == 1
    assert len(numbers) == 9
    assert frc[0,0] == 0.000014646
    assert frc[1,-1] == 0.005043566
    assert frc[-1,1] == 0.002557226
    assert frc.shape == (9, 3)


def test_scan_time():
    fn_log = pkg_resources.resource_filename(__name__, '../../data/test/gaussian_sioh4_md.log')
    with open(fn_log) as f:
        time, step, ekin, epot, etot = _scan_g09_time(f)
        assert time == 0.0
        assert step == 2
        assert ekin == 0.0306188
        assert epot == -592.9048374
        assert etot == -592.8742186

        time, step, ekin, epot, etot = _scan_g09_time(f)
        assert time == 1.125278*femtosecond
        assert step == 3
        assert ekin == 0.0244215
        assert epot == -592.8986401
        assert etot == -592.8742186


def test_scan_pos_vel():
    vel_unit = np.sqrt(amu)/second
    fn_log = pkg_resources.resource_filename(__name__, '../../data/test/gaussian_sioh4_md.log')
    with open(fn_log) as f:
        _scan_to_line(f, " Cartesian coordinates: (bohr)") # skip first one, has different format
        pos, vel = _scan_g09_pos_vel(f)
        assert pos[0,0] == -1.287811626725E-02
        assert pos[-1,-1] == 2.710579145562E+00
        assert pos.shape == (9, 3)
        assert vel[1, 0] == 5.750552889614E+13*vel_unit
        assert vel[-2, 2] == 1.741570818851E+13*vel_unit
        assert vel.shape == (9, 3)


def test_to_hdf():
    vel_unit = np.sqrt(amu)/second
    fn_xyz = pkg_resources.resource_filename(__name__, '../../data/test/gaussian_sioh4_md.xyz')
    fn_log = pkg_resources.resource_filename(__name__, '../../data/test/gaussian_sioh4_md.log')
    with h5.File('yaff.conversion.test.test_gaussian.test_to_hdf5.h5', driver='core', backing_store=False) as f:
        system = System.from_file(fn_xyz)
        system.to_hdf5(f)
        # Actual trajectory conversion, twice
        for i in xrange(2):
            offset = 2*i
            g09log_to_hdf5(f, fn_log)
            assert 'trajectory' in f
            assert get_last_trajectory_row(f['trajectory']) == 2+offset
            assert 'pos' in f['trajectory']
            assert f['trajectory/pos'].shape == (2+offset, 9, 3)
            assert f['trajectory/pos'][offset,0,0] == -1.287811626725E-02
            assert f['trajectory/pos'][-1,-1,-1] == 2.710239686065E+00
            assert 'vel' in f['trajectory']
            assert f['trajectory/vel'].shape == (2+offset, 9, 3)
            assert f['trajectory/vel'][offset,0,0] == -6.493457131863E+13*vel_unit
            assert f['trajectory/vel'][-1,-1,-1] == 4.186482857132E+12*vel_unit
            assert 'frc' in f['trajectory']
            assert f['trajectory/frc'].shape == (2+offset, 9, 3)
            assert f['trajectory/frc'][offset,0,0] == 0.002725302
            assert f['trajectory/frc'][-1,-1,-1] == 0.008263482
            assert 'time' in f['trajectory']
            assert f['trajectory/time'].shape == (2+offset, 1)
            assert f['trajectory/time'][offset] == 0.0
            assert f['trajectory/time'][-1] == 1.125278*femtosecond
            assert 'step' in f['trajectory']
            assert f['trajectory/step'].shape == (2+offset, 1)
            assert f['trajectory/step'][offset] == 2
            assert f['trajectory/step'][-1] == 3
            assert 'epot' in f['trajectory']
            assert f['trajectory/epot'].shape == (2+offset, 1)
            assert f['trajectory/epot'][offset] == -592.9048374
            assert f['trajectory/epot'][-1] == -592.8986401
            assert 'ekin' in f['trajectory']
            assert f['trajectory/ekin'].shape == (2+offset, 1)
            assert f['trajectory/ekin'][offset] == 0.0306188
            assert f['trajectory/ekin'][-1] == 0.0244215
            assert 'etot' in f['trajectory']
            assert f['trajectory/etot'].shape == (2+offset, 1)
            assert f['trajectory/etot'][offset] == -592.8742186
            assert f['trajectory/etot'][-1] == -592.8742186
        f.close()
