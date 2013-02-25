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

from yaff import *
from yaff.sampling.test.common import get_ff_bks

def test_check_delta_cartesian():
    dof = CartesianDOF(get_ff_bks())
    dof.check_delta()


def test_check_delta_cartesian_partial():
    dof = CartesianDOF(get_ff_bks(), select=[0, 1, 2])
    dof.check_delta()


def test_check_delta_full_cell():
    ff = get_ff_bks()
    dof = FullCellDOF(ff)
    dof.check_delta()
    zero = np.zeros(dof.ndof, dtype=bool)
    zero[:9] = True
    dof.check_delta(zero=zero)
    dof.check_delta(zero=~zero)
    dof = FullCellDOF(ff, do_frozen=True)
    dof.check_delta()


def test_check_delta_iso_cell():
    ff = get_ff_bks()
    dof = IsoCellDOF(ff)
    dof.check_delta()
    zero = np.zeros(dof.ndof, dtype=bool)
    zero[:1] = True
    dof.check_delta(zero=zero)
    dof.check_delta(zero=~zero)
    dof = IsoCellDOF(ff, do_frozen=True)
    dof.check_delta()


def test_check_delta_aniso_cell():
    ff = get_ff_bks()
    dof = AnisoCellDOF(ff)
    dof.check_delta()
    zero = np.zeros(dof.ndof, dtype=bool)
    zero[:3] = True
    dof.check_delta(zero=zero)
    dof.check_delta(zero=~zero)
    dof = AnisoCellDOF(ff, do_frozen=True)
    dof.check_delta()
