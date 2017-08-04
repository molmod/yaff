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
#!/usr/bin/env python
"""Script to check for incorrect whitespace in a list of files."""

from __future__ import print_function

import sys

retcode = 0
for fn in sys.argv[1:]:
    with open(fn) as f:
        lines = f.readlines()
    bad = False
    for line in lines:
        if line.endswith(' \n'):
            bad = True
            break
        if '\t' in line:
            bad = True
            break
    bad |= (lines[-1] == '\n')
    if bad:
        print('Whitespace errors in {}'.format(fn))
        retcode = 1
sys.exit(retcode)
