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
#--

import numpy as np
import sys

# import the yaff library
from yaff import *

# Control the amount of screen output.
log.set_level(log.silent)

__all__ = ['make_system','make_forcefield']

def make_system(nx=1, ny=1, nz=1):
    '''
    Make a system with nx repetitions along first, ny along second and nz along
    third cell vector
    '''
    # Load system from file
    system = System.from_file('system.chk')
    # Set the charges
    system.charges = np.zeros(system.numbers.shape)
    system.charges[system.numbers==14] = 2.4
    system.charges[system.numbers==8] = -1.2
    # Make a supercell
    system = system.supercell(nx, ny, nz)
    return system

def make_forcefield(nx=1, ny=1, nz=1):
    # Construct the system
    system = make_system(nx, ny, nz)
    # Generate the force field
    ff = ForceField.generate(system, 'parameters.txt')
    return ff
