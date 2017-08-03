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


import tempfile, shutil, os, numpy as np

from yaff import *
from molmod.test.common import tmpdir


def test_blav():
    # generate a time-correlated random signal
    n = 50000
    eps0 = 30.0/n
    eps1 = 1.0
    y = np.sin(np.random.normal(0, eps0, n).cumsum() + np.random.normal(0, eps1, n))
    # create a temporary directory to write the plot to
    with tmpdir(__name__, 'test_blav') as dn:
        fn_png = '%s/blav.png' % dn
        error, sinef = blav(y, 100, fn_png)
        assert os.path.isfile(fn_png)
