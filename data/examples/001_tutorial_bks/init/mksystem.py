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


# This is a python script! Lines like these, starting with a '#' are ignored
# in the python language and are used to explain what this script does.
# All other lines are simply executed in order. Variables are assigned as
# follows:
#    a = 2.0
#    b = 5.0
# Simple arithmetic operations are carried out as follows:
#    c = (a+b)/(5.1*b)
# Everything can be printed on screen, in case you want to know an intermediate
# result:
#    print 'Hello world,'
#    print 'The value of c is', c
# The official Python tutorial, see http://docs.python.org/3/tutorial/, is a
# nice way to learn Python. However, for this session, there is no need to do
# this.

# First the the numpy library is imported. Numpy is great for efficient
# operations on arrays of (numerical) data. It is used throughout Yaff. Whenever
# you see np.xxx, some feature of the Numpy library is used. More information
# about Numpy can be found here: http://numpy.org/
# (There is no need to go through the Numpy documentation for this hands-on
# session.)
import numpy as np

# Then everything from the Yaff library is imported. All variables, classes,
# constants, functions, ... defined in the Yaff library are now accessible in
# this script. The Yaff documentation can be found here:
# http://molmod.github.com/yaff/
from yaff import *

# Definition of atom types. Each row corresponds to a tuple with (i) the name of
# the atom type and (ii) an ATSELECT rule that defines the atom type formally.
# ATSELECT is a one-line language similar to SMARTS. If you want to know more
# about it: http://molmod.github.com/yaff/ug_atselect.html
# (There is no need to change the atom types. They are used in the file bks.pot)
ffatype_rules = [
    ('Si_4', '14'),
    ('O_B', '8'),
]

# Load the real-space cell vectors from the file rvecs.txt. The file contains
# data in angstrom, so the part '*angstrom' is used to convert the numbers to
# the internal Yaff units. (Internally, atomic units are used.)
rvecs = np.loadtxt('rvecs.txt')*angstrom

# Create a system object, based on the nuclear coordinates in struct.xyz and
# the cell vectors loaded in the previous line.
system = System.from_file('struct.xyz', rvecs=rvecs)

# Assign default atomic masses.
system.set_standard_masses()

# Detect the atom types
system.detect_ffatypes(ffatype_rules)

# Optionally, if you like, create a 2x2x2 super cell. When this line is
# uncommented, the system variable is overwritten by its super cell.
#system = system.supercell(2,2,2)

# Finally write the system to a file init.chk. This is a text-based checkpoint
# file that will be used as the initial state for the script opt/simulation.py.
system.to_file('init.chk')
