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
    'check_gpos_cv_fd', 'check_vtens_cv_fd', 'get_part_water32_9A_lj',
    'check_nlow_nhigh_part'
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


def check_gpos_cv_fd(cv, delta=1e-4, threshold=1e-7):
    '''
    Check gpos of CollectiveVariable using finite differences

        **Arguments:**

        cv
            A CollectiveVariable instance

        **Optional arguments:**

        delta
            Deviation used in the finite difference scheme

        threshold
            Measure for the allowed error
    '''
    # Reference state
    gpos = np.zeros((cv.system.natom, 3), float)
    pos_orig = cv.system.pos.copy()
    cv0 = cv.compute(gpos=gpos)
    # Finite difference deviations and weights
    stencil = [(-3,-1.0/60.),(-2,3.0/20.0),(-1,-3.0/4.0),(1,3.0/4.0),(2,-3.0/20.0),(3,1.0/60.0)]
    gpos_fd = np.zeros((cv.system.natom,3))
    for iatom in range(cv.system.natom):
        for alpha in range(3):
            deriv = 0.0
            for dev, weight in stencil:
                cv.system.pos[:] = pos_orig.copy()
                cv.system.pos[iatom,alpha] += dev*delta
                cv_value = cv.compute()
                deriv += weight*cv_value/delta
            gpos_fd[iatom,alpha] = deriv
    # Compare analytical and numerical derivative
    maxdev = np.amax(np.abs(gpos_fd-gpos))
    rmsd = np.std(gpos_fd-gpos)
    ref = np.std(gpos)
    # If all forces are zero, we use the threshold itself;
    # Else, we use the threshold times the RMSD of the forces
    if ref==0.0: ref = 1.0
#    print("Max deviation: %8.1e (%8.1e) | RMSD deviation: %8.1e (%8.1e)"%
#        (maxdev,ref*threshold,rmsd,ref*threshold))
    assert maxdev<ref*threshold
    assert rmsd<ref*threshold


def check_vtens_cv_fd(cv, delta=1e-4, threshold=1e-7):
    '''
    Check vtens of CollectiveVariable using finite differences

        **Arguments:**

        cv
            A CollectiveVariable instance

        **Optional arguments:**

        delta
            Deviation used in the finite difference scheme

        threshold
            Measure for the allowed error
    '''
    # Reference state
    rvecs_orig = cv.system.cell.rvecs.copy()
    pos_orig = cv.system.pos.copy()
    vtens = np.zeros((3, 3), float)
    cv0 = cv.compute(vtens=vtens)
    gvecs = np.linalg.inv(rvecs_orig).transpose()
    reduced = np.dot(cv.system.pos, cv.system.cell.gvecs.transpose()).copy()
    # Finite difference deviations and weights
    stencil = [(-3,-1.0/60.),(-2,3.0/20.0),(-1,-3.0/4.0),(1,3.0/4.0),(2,-3.0/20.0),(3,1.0/60.0)]
    vtens_fd = np.zeros((3,3))
    for alpha in range(3):
        for beta in range(3):
            deriv = 0.0
            for dev, weight in stencil:
                rvecs = rvecs_orig.copy()
                rvecs[alpha,beta] += dev*delta
                cv.system.cell.update_rvecs(rvecs)
                cv.system.pos[:] = np.dot(reduced, rvecs)
                cv_value = cv.compute()
                deriv += weight*cv_value/delta
            vtens_fd[alpha,beta] = deriv
    vtens_fd = np.dot(rvecs_orig.T,vtens_fd)
    # Compare analytical and numerical derivative
    maxdev = np.amax(np.abs(vtens_fd-vtens))
    rmsd = np.std(vtens_fd-vtens)
    ref = np.std(vtens)
#    print("Max deviation: %8.1e (%8.1e) | RMSD deviation: %8.1e (%8.1e)"%
#        (maxdev,ref*threshold,rmsd,ref*threshold))
    assert maxdev<ref*threshold
    assert rmsd<ref*threshold


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


def check_nlow_nhigh_part(system, part_generator, nlow, nhigh, **kwargs):
    '''Check that nlow and nhigh exclude the correct interactions'''
    # 0) The full system
    part0 = part_generator(system, **kwargs)
    gpos0 = np.zeros((system.natom,3))
    vtens0 = np.zeros((3,3))
    e0 = part0.compute(gpos0, vtens0)
    # 1) The system with only atoms whose index is smaller than or equal to
    # nlow
    system1 = system.subsystem(np.arange(nlow))
    part1 = part_generator(system1, **kwargs)
    gpos1 = np.zeros((system1.natom,3))
    vtens1 = np.zeros((3,3))
    e1 = part1.compute(gpos1, vtens1)
    # 2) The system with only atoms whose index is larger than nhigh (take for
    # nhigh = -1
    if nhigh==-1:
        system2 = system.subsystem(np.arange(system.natom,system.natom))
    else:
        system2 = system.subsystem(np.arange(nhigh,system.natom))
    part2 = part_generator(system2, **kwargs)
    gpos2 = np.zeros((system2.natom,3))
    vtens2 = np.zeros((3,3))
    e2 = part2.compute(gpos2, vtens2)
    # 3) Direct computation excluding the requested pairs
    kwargs['nlow'] = nlow
    kwargs['nhigh'] = nhigh
    part3 = part_generator(system, **kwargs)
    gpos3 = np.zeros((system.natom,3))
    vtens3 = np.zeros((3,3))
    e3 = part3.compute(gpos3, vtens3)
    # Energy of 3) should equal energy of 0) minus 1) minus 2)
    print("E0 = %20.12f E1 = %20.12f E2 = %20.12f | E0-E1-E2 = %20.12f deltaE = %20.12f" %
        (e0,e1,e2,e0-e1-e2,e3))
    assert np.abs(e0-e1-e2-e3)<1e-10
    # Same for forces, but pay attention to attribute forces to correct atoms
    gpos0[np.arange(nlow)] -= gpos1
    if nhigh != -1:
        gpos0[np.arange(nhigh,system.natom)] -= gpos2
    assert np.all( np.abs(gpos0-gpos3) < 1e-10 )
    # Same for vtens
    assert np.all( np.abs(vtens0-vtens1-vtens2-vtens3) < 1e-10 )
