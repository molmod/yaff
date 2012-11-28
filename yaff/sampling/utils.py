# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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


from molmod import boltzmann

import numpy as np


__all__ = [
    'get_random_vel', 'remove_com_vel',
]


def get_random_vel(temp0, scalevel0, masses, select=None):
    if select is not None:
        masses = masses[select]
    shape = len(masses), 3
    result = np.random.normal(0, 1, shape)*np.sqrt(boltzmann*temp0/masses).reshape(-1,1)
    if scalevel0 and temp0 > 0:
        temp = (result**2*masses.reshape(-1,1)).mean()/boltzmann
        scale = np.sqrt(temp0/temp)
        result *= scale
    return result


def remove_com_vel(vel, masses):
    # compute the center of mass velocity
    com_vel = np.dot(masses, vel)/masses.sum()
    # subtract
    vel[:] -= com_vel
