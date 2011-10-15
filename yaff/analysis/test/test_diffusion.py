# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
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

import shutil, os, h5py

from yaff import *
from yaff.analysis.test.common import get_nve_water32


def test_rdf1_offline():
    dn_tmp, nve, f = get_nve_water32()
    try:
        select = nve.ff.system.get_indexes('O')
        diff = Diffusion(f, select=select)
        assert 'trajectory/pos_diff' in f
        assert 'trajectory/pos_diff/msds' in f
        assert 'trajectory/pos_diff/time' in f
        assert 'trajectory/pos_diff/msdcounters' in f
        assert 'trajectory/pos_diff/msdsums' in f
        assert 'trajectory/pos_diff/pars' in f
        fn_png = '%s/msds.png' % dn_tmp
        diff.plot(fn_png)
        assert os.path.isfile(fn_png)
    finally:
        shutil.rmtree(dn_tmp)
        f.close()
