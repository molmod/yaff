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


import numpy as np

from yaff import *
from yaff.sampling.test.common import get_ff_water32, get_ff_water, get_ff_bks


def test_hessian_partial_water32():
    ff = get_ff_water32()
    select = [1, 2, 3, 14, 15, 16]
    hessian = estimate_cart_hessian(ff, select=select)
    assert hessian.shape == (18, 18)


def test_hessian_full_water():
    ff = get_ff_water()
    hessian = estimate_cart_hessian(ff)
    assert hessian.shape == (9, 9)
    evals = np.linalg.eigvalsh(hessian)
    print evals
    assert sum(abs(evals) < 1e-10) == 3


def test_elastic_water32():
    ff = get_ff_water32()
    elastic = estimate_elastic(ff, do_frozen=True)
    assert elastic.shape == (6, 6)


def test_bulk_elastic_bks():
    ff = get_ff_bks()
    elastic = estimate_elastic(ff)
    assert elastic.shape == (6, 6)
