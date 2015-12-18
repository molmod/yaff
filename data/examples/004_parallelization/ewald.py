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

import sys

from tools import *

assert len(sys.argv)==5

# Read command-line arguments
nx, ny, nz = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
nrep = int(sys.argv[4])

# Construct a complete force field
ff = make_forcefield(nx,ny,nz)

# Select reciprocal Ewald part
ewald = None
for part in ff.parts:
    if part.name=='ewald_reci':
        ewald = part
assert ewald is not None

# Do the computation nrep times
for i in xrange(nrep):
    ewald.compute()
