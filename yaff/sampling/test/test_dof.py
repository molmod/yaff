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

import numpy as np

from molmod.minimizer import check_delta
from yaff import *
from yaff.sampling.test.common import get_ff_bks, get_ff_graphene, \
    get_ff_polyethylene, get_ff_nacl


def test_delta_cartesian():
    dof = CartesianDOF(get_ff_bks())
    dof.check_delta()


def test_delta_cartesian_partial():
    dof = CartesianDOF(get_ff_bks(), select=[0, 1, 2])
    dof.check_delta()


def check_delta_cell(ff, DOFClass, kwargs):
    dof = DOFClass(ff, **kwargs)
    dof.check_delta()
    zero = np.zeros(dof.ndof, dtype=bool)
    zero[:dof.ncelldof] = True
    dof.check_delta(zero=zero)
    dof.check_delta(zero=~zero)
    kwargs['do_frozen'] = True
    dof = DOFClass(ff, **kwargs)
    dof.check_delta()


def check_cell_jacobian(ff, DOFClass, kwargs):
    kwargs['do_frozen'] = True
    dof = DOFClass(ff, **kwargs)

    def fun(celldofs, do_gradient=False):
        rvecs = dof._cellvars_to_rvecs(dof._expand_celldofs(celldofs))
        if do_gradient:
            dof._update(celldofs)
            return rvecs.ravel(), dof._get_celldofs_jacobian(celldofs)
        else:
            return rvecs.ravel()

    x = dof._reduce_cellvars(dof.cellvars0)
    eps = 1e-5
    dxs = np.random.uniform(-eps, eps, (100, len(x)))
    check_delta(fun, x, dxs)


def check_ff(ff, DOFClass, ncellvar, kwargs):
    check_cell_jacobian(ff, DOFClass, kwargs)
    check_delta_cell(ff, DOFClass, kwargs)
    if ncellvar > 1:
        freemask = np.ones(ncellvar, dtype=bool)
        freemask[1::2] = False
        kwargs['freemask'] = freemask
        check_cell_jacobian(ff, DOFClass, kwargs)
        check_delta_cell(ff, DOFClass, kwargs)


def test_full_3d():
    check_ff(get_ff_bks(), FullCellDOF, 9, {})

def test_full_2d():
    check_ff(get_ff_graphene(), FullCellDOF, 6, {})

def test_full_1d():
    check_ff(get_ff_polyethylene(), FullCellDOF, 3, {})


def test_strain_3d():
    check_ff(get_ff_bks(), StrainCellDOF, 6, {})

def test_strain_2d():
    check_ff(get_ff_graphene(), StrainCellDOF, 3, {})

def test_strain_1d():
    check_ff(get_ff_polyethylene(), StrainCellDOF, 1, {})


def test_aniso_3d():
    check_ff(get_ff_bks(), AnisoCellDOF, 3, {})

def test_aniso_2d():
    check_ff(get_ff_graphene(), AnisoCellDOF, 2, {})

def test_aniso_1d():
    check_ff(get_ff_polyethylene(), AnisoCellDOF, 1, {})


def test_iso_3d():
    check_ff(get_ff_bks(), IsoCellDOF, 1, {})

def test_iso_2d():
    check_ff(get_ff_graphene(), IsoCellDOF, 1, {})

def test_iso_1d():
    check_ff(get_ff_polyethylene(), IsoCellDOF, 1, {})


def test_fixedbc_3d():
    check_ff(get_ff_bks(align_ax=True), FixedBCDOF, 1, {})


def test_fixedvolortho_3d():
    check_ff(get_ff_nacl(), FixedVolOrthoCellDOF, 1, {})
