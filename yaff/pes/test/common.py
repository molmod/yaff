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

from molmod import check_delta

from yaff import *
from yaff.test.common import get_system_water32


__all__ = [
    'check_gpos_part', 'check_vtens_part', 'check_gpos_ff', 'check_vtens_ff',
    'get_part_water32_9A_lj'
]


def check_gpos_part(system, part, nlists=None):
    def fn(x, do_gradient=False):
        system.pos[:] = x.reshape(system.natom, 3)
        if nlists is not None:
            nlists.update()
        if do_gradient:
            gpos = np.zeros(system.pos.shape, float)
            e = part.compute(gpos)
            assert np.isfinite(e)
            assert np.isfinite(gpos).all()
            return e, gpos.ravel()
        else:
            e = part.compute()
            assert np.isfinite(e)
            return e

    x = system.pos.ravel()
    dxs = np.random.normal(0, 1e-4, (100, len(x)))
    check_delta(fn, x, dxs)


def check_vtens_part(system, part, nlists=None, symm_vtens=True):
    '''
        * symm_vtens: Check if the virial tensor is a symmetric matrix.
                      For instance for dipole interactions, this is not true
    '''
    # define some rvecs and gvecs
    if system.cell.nvec == 3:
        gvecs = system.cell.gvecs
        rvecs = system.cell.rvecs
    else:
        gvecs = np.identity(3, float)
        rvecs = np.identity(3, float)

    # Get the reduced coordinates
    reduced = np.dot(system.pos, gvecs.transpose())
    if symm_vtens:
        assert abs(np.dot(reduced, rvecs) - system.pos).max() < 1e-10

    def fn(x, do_gradient=False):
        rvecs = x.reshape(3, 3)
        if system.cell.nvec == 3:
            system.cell.update_rvecs(rvecs)
        system.pos[:] = np.dot(reduced, rvecs)
        if nlists is not None:
            nlists.update()
        if do_gradient:
            vtens = np.zeros((3, 3), float)
            e = part.compute(vtens=vtens)
            gvecs = np.linalg.inv(rvecs).transpose()
            grvecs = np.dot(gvecs, vtens)
            assert np.isfinite(e)
            assert np.isfinite(vtens).all()
            assert np.isfinite(grvecs).all()
            if symm_vtens:
                assert abs(vtens - vtens.transpose()).max() < 1e-10
            return e, grvecs.ravel()
        else:
            e = part.compute()
            assert np.isfinite(e)
            return e

    x = rvecs.ravel()
    dxs = np.random.normal(0, 1e-4, (100, len(x)))
    check_delta(fn, x, dxs)


def check_gpos_ff(ff):
    def fn(x, do_gradient=False):
        ff.update_pos(x.reshape(ff.system.natom, 3))
        if do_gradient:
            gpos = np.zeros(ff.system.pos.shape, float)
            e = ff.compute(gpos)
            assert np.isfinite(e)
            assert np.isfinite(gpos).all()
            return e, gpos.ravel()
        else:
            e = ff.compute()
            assert np.isfinite(e)
            return e

    x = ff.system.pos.ravel()
    dxs = np.random.normal(0, 1e-4, (100, len(x)))
    check_delta(fn, x, dxs)


def check_vtens_ff(ff):
    # define some rvecs and gvecs
    if ff.system.cell.nvec == 3:
        gvecs = ff.system.cell.gvecs
        rvecs = ff.system.cell.rvecs
    else:
        gvecs = np.identity(3, float)
        rvecs = np.identity(3, float)

    # Get the reduced coordinates
    reduced = np.dot(ff.system.pos, gvecs.transpose())
    assert abs(np.dot(reduced, rvecs) - ff.system.pos).max() < 1e-10

    def fn(x, do_gradient=False):
        rvecs = x.reshape(3, 3)
        if ff.system.cell.nvec == 3:
            ff.update_rvecs(rvecs)
        ff.update_pos(np.dot(reduced, rvecs))
        if do_gradient:
            vtens = np.zeros((3, 3), float)
            e = ff.compute(vtens=vtens)
            gvecs = np.linalg.inv(rvecs).transpose()
            grvecs = np.dot(gvecs, vtens)
            assert np.isfinite(e)
            assert np.isfinite(vtens).all()
            assert np.isfinite(grvecs).all()
            assert abs(vtens - vtens.transpose()).max() < 1e-10
            return e, grvecs.ravel()
        else:
            e = ff.compute()
            assert np.isfinite(e)
            return e

    x = rvecs.ravel()
    dxs = np.random.normal(0, 1e-4, (100, len(x)))
    check_delta(fn, x, dxs)


def get_part_water32_9A_lj():
    # Initialize system, nlist and scaling
    system = get_system_water32()
    nlist = NeighborList(system)
    scalings = Scalings(system)
    # Initialize parameters
    rminhalf_table = {1: 0.2245*angstrom, 8: 1.7682*angstrom}
    epsilon_table = {1: -0.0460*kcalmol, 8: -0.1521*kcalmol}
    sigmas = np.zeros(96, float)
    epsilons = np.zeros(96, float)
    for i in range(system.natom):
        sigmas[i] = rminhalf_table[system.numbers[i]]*(2.0)**(5.0/6.0)
        epsilons[i] = epsilon_table[system.numbers[i]]
    # Create the pair_pot and part_pair
    rcut = 9*angstrom
    pair_pot = PairPotLJ(sigmas, epsilons, rcut, Hammer(1.0))
    assert abs(pair_pot.sigmas - sigmas).max() == 0.0
    assert abs(pair_pot.epsilons - epsilons).max() == 0.0
    part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
    # Create a pair function:
    def pair_fn(i, j, d, delta):
        sigma = 0.5*(sigmas[i]+sigmas[j])
        epsilon = np.sqrt(epsilons[i]*epsilons[j])
        x = (sigma/d)**6
        return 4*epsilon*(x*(x-1))*np.exp(1.0/(d-rcut))
    return system, nlist, scalings, part_pair, pair_fn
