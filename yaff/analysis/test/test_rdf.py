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

import shutil, os

from yaff import *
from yaff.analysis.test.common import get_nve_water32


def test_rdf1_offline():
    dn_tmp, nve, f = get_nve_water32()
    try:
        select = nve.ff.system.get_indexes('O')
        rdf = RDF(f, rcut=4.5*angstrom, rspacing=0.1*angstrom, select0=select)
        assert 'trajectory/pos_rdf' in f
        assert 'trajectory/pos_rdf/d' in f
        assert 'trajectory/pos_rdf/counts' in f
        assert 'trajectory/pos_rdf/rdf' in f
        assert 'trajectory/pos_rdf/crdf' in f
        fn_png = '%s/rdf.png' % dn_tmp
        rdf.plot(fn_png)
        assert os.path.isfile(fn_png)
        fn_png = '%s/crdf.png' % dn_tmp
        rdf.plot_crdf(fn_png)
        assert os.path.isfile(fn_png)
    finally:
        shutil.rmtree(dn_tmp)
        f.close()


def test_rdf2_offline():
    dn_tmp, nve, f = get_nve_water32()
    try:
        select0 = nve.ff.system.get_indexes('O')
        select1 = nve.ff.system.get_indexes('H')
        rdf = RDF(f, rcut=4.5*angstrom, rspacing=0.1*angstrom, select0=select0, select1=select1)
        assert 'trajectory/pos_rdf' in f
        assert 'trajectory/pos_rdf/d' in f
        assert 'trajectory/pos_rdf/counts' in f
        assert 'trajectory/pos_rdf/rdf' in f
        assert 'trajectory/pos_rdf/crdf' in f
        fn_png = '%s/rdf.png' % dn_tmp
        rdf.plot(fn_png)
        assert os.path.isfile(fn_png)
    finally:
        shutil.rmtree(dn_tmp)
        f.close()
