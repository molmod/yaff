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
import pkg_resources

from yaff import *


def test_simple_transform():
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    pf0 = Parameters.from_file(fn_pars)
    rules = [ScaleRule('BONDFUES', 'PARS', 'O\s*H', 3)]
    mods = [ParameterModifier(rules)]
    pt = ParameterTransform(pf0, mods)
    assert (pt.get_init() == np.array([1.0])).all()
    pf1 = pt([2.0])
    assert pf1['BONDFUES']['PARS'][0][1] == 'O H 4.0088096730e+03  2.0476480000e+00'
