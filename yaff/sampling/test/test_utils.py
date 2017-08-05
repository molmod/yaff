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

import numpy as np

from yaff import *

from yaff.test.common import get_system_water32, get_system_glycine, \
    get_system_graphene8


def test_remove_com_moment():
    sys = get_system_water32()
    sys.set_standard_masses()
    masses = sys.masses
    vel = get_random_vel(300, False, masses)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() > 1e-10
    remove_com_moment(vel, masses)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() < 1e-10


def test_remove_angular_moment():
    sys = get_system_glycine()
    sys.set_standard_masses()
    masses = sys.masses
    vel = get_random_vel(300, False, masses)
    remove_com_moment(vel, masses)
    ang_mom = angular_moment(sys.pos, vel, masses)
    assert abs(ang_mom).max() > 1e-10
    remove_angular_moment(sys.pos, vel, masses)
    ang_mom = angular_moment(sys.pos, vel, masses)
    assert abs(ang_mom).max() < 1e-10


def test_clean_momenta_3d():
    sys = get_system_water32()
    sys.set_standard_masses()
    masses = sys.masses
    vel = get_random_vel(300, False, masses)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() > 1e-10
    clean_momenta(sys.pos, vel, masses, sys.cell)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() < 1e-10


def test_clean_momenta_2d():
    sys = get_system_graphene8()
    sys.set_standard_masses()
    masses = sys.masses
    vel = get_random_vel(300, False, masses)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() > 1e-10
    clean_momenta(sys.pos, vel, masses, sys.cell)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() < 1e-10


def test_clean_momenta_0d():
    sys = get_system_glycine()
    sys.set_standard_masses()
    masses = sys.masses
    vel = get_random_vel(300, False, masses)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() > 1e-10
    ang_mom = angular_moment(sys.pos, vel, masses)
    assert abs(ang_mom).max() > 1e-10
    clean_momenta(sys.pos, vel, masses, sys.cell)
    com_mom = np.dot(masses, vel)
    assert abs(com_mom).max() < 1e-10
    ang_mom = angular_moment(sys.pos, vel, masses)
    assert abs(ang_mom).max() < 1e-10
