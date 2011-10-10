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
from yaff.analysis.test.common import get_water_32_simulation


def test_spectrum_offline():
    dn_tmp, nve, f = get_water_32_simulation()
    try:
        for bsize in 2, 4, 5:
            spectrum = Spectrum(f, bsize=bsize)
            fn_png = '%s/spectrum%i.png' % (dn_tmp, bsize)
            spectrum.plot(fn_png)
            assert os.path.isfile(fn_png)
            fn_png = '%s/ac%i.png' % (dn_tmp, bsize)
            spectrum.plot_ac(fn_png)
            assert os.path.isfile(fn_png)
    finally:
        shutil.rmtree(dn_tmp)
