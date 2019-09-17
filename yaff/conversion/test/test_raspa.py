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
from __future__ import print_function

import pkg_resources

from yaff.conversion.raspa import *
from molmod.units import kelvin, pascal

def test_raspa_read_loading():
    fn = pkg_resources.resource_filename(__name__,
            '../../data/test/output_MIL53_2.2.2_298.000000_10000.data')
    T, P, fugacity, N, Nerr = read_raspa_loading(fn)
    assert T==298*kelvin
    assert P==10000*pascal
    assert fugacity==9994.52870926490505*pascal
    assert N==0.2916750000
    assert Nerr==0.0178022822
