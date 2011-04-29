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
cimport numpy as np
cimport cell
cimport nlists
cimport pair_pot
cimport ewald
cimport dlist
cimport iclist
cimport vlist


__all__ = [
    'nlist_status_init', 'nlist_update', 'nlist_status_finish',
    'PairPot', 'PairPotLJ', 'PairPotEI', 'compute_ewald_reci',
    'compute_ewald_corr', 'dlist_forward', 'dlist_back', 'iclist_forward',
    'iclist_back', 'vlist_forward', 'vlist_back',
]


#
# Cell
#

cdef class Cell:
    cdef cell.cell_type* _c_cell

    def __cinit__(self, *args, **kwargs):
        self._c_cell = cell.cell_new()
        if self._c_cell is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_cell is not NULL:
            cell.cell_free(self._c_cell)

    def __init__(self, np.ndarray[double, ndim=2] rvecs):
        self.update_rvecs(rvecs)

    def update_rvecs(self, np.ndarray[double, ndim=2] rvecs):
        cdef np.ndarray[double, ndim=2] mod_rvecs
        cdef np.ndarray[double, ndim=2] gvecs
        cdef int nvec
        if rvecs.size == 0:
            mod_rvecs = np.identity(3, float)
            gvecs = mod_rvecs
            nvec = 0
        else:
            if not rvecs.ndim==2 or not rvecs.flags['C_CONTIGUOUS'] or rvecs.shape[0] > 3 or rvecs.shape[1] != 3:
                raise TypeError('rvecs must be a C contiguous array with three columns and at most three rows.')
            nvec = len(rvecs)
            Up, Sp, Vt = np.linalg.svd(rvecs, full_matrices=True)
            S = np.ones(3, float)
            S[:nvec] = Sp
            U = np.identity(3, float)
            U[:nvec,:nvec] = Up
            mod_rvecs = np.dot(U*S, Vt)
            mod_rvecs[:nvec] = rvecs
            gvecs = np.dot(U/S, Vt)
        cell.cell_update(self._c_cell, <double*>mod_rvecs.data, <double*>gvecs.data, nvec)

    def get_nvec(self):
        return cell.cell_get_nvec(self._c_cell)

    nvec = property(get_nvec)

    def get_volume(self):
        return cell.cell_get_volume(self._c_cell)

    volume = property(get_volume)

    def get_rvecs(self, full=False):
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_rvecs(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rvecs = property(get_rvecs)

    def get_gvecs(self, full=False):
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_gvecs(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gvecs = property(get_gvecs)

    def get_rspacings(self, full=False):
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_rspacings(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rspacings = property(get_rspacings)

    def get_gspacings(self, full=False):
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_gspacings(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gspacings = property(get_gspacings)

    def mic(self, np.ndarray[double, ndim=1] delta):
        assert delta.size == 3
        cell.cell_mic(<double *>delta.data, self._c_cell)


#
# Neighbor lists
#


def nlist_status_init(center_index, rmax):
    # five integer status fields:
    # * r0
    # * r1
    # * r2
    # * other_index
    # * number of rows consumed
    result = np.array([0, 0, 0, 0, 0], int)
    for i in xrange(len(rmax)):
        if len(rmax) > 0:
            result[i] = -rmax[i]
    return result


def nlist_update(np.ndarray[double, ndim=2] pos, long center_index,
                 double cutoff, np.ndarray[long, ndim=1] rmax,
                 Cell unitcell, np.ndarray[long, ndim=1] nlist_status,
                 np.ndarray[nlists.nlist_row_type, ndim=1] nlist):
    assert pos.shape[1] == 3
    assert pos.flags['C_CONTIGUOUS']
    assert center_index >= 0
    assert center_index < pos.shape[0]
    assert cutoff > 0
    assert rmax.shape[0] <= 3
    assert rmax.flags['C_CONTIGUOUS']
    assert nlist_status.shape[0] == 5
    assert nlist_status.flags['C_CONTIGUOUS']
    assert nlist.flags['C_CONTIGUOUS']
    assert rmax.shape[0] == unitcell.nvec
    return nlists.nlist_update_low(
        <double*>pos.data, center_index, cutoff, <long*>rmax.data,
        unitcell._c_cell, <long*>nlist_status.data,
        <nlists.nlist_row_type*>nlist.data, len(pos), len(nlist)
    )


def nlist_status_finish(nlist_status):
    return nlist_status[4]


#
# Pair potentials
#


cdef class PairPot:
    cdef pair_pot.pair_pot_type* _c_pair_pot

    def __cinit__(self, *args, **kwargs):
        self._c_pair_pot = pair_pot.pair_pot_new()
        if self._c_pair_pot is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if pair_pot.pair_pot_ready(self._c_pair_pot):
            pair_pot.pair_data_free(self._c_pair_pot)
        if self._c_pair_pot is not NULL:
            pair_pot.pair_pot_free(self._c_pair_pot)

    def get_cutoff(self):
        return pair_pot.pair_pot_get_cutoff(self._c_pair_pot)

    cutoff = property(get_cutoff)

    def get_smooth(self):
        return pair_pot.pair_pot_get_smooth(self._c_pair_pot)

    smooth = property(get_smooth)

    def compute(self, long center_index,
                np.ndarray[nlists.nlist_row_type, ndim=1] nlist,
                np.ndarray[pair_pot.scaling_row_type, ndim=1] scaling,
                np.ndarray[double, ndim=2] gpos,
                np.ndarray[double, ndim=2] vtens):
        cdef double *my_gpos
        cdef double *my_vtens

        assert pair_pot.pair_pot_ready(self._c_pair_pot)
        assert nlist.flags['C_CONTIGUOUS']
        assert scaling.flags['C_CONTIGUOUS']

        if gpos is None:
            my_gpos = NULL
        else:
            assert gpos.flags['C_CONTIGUOUS']
            assert gpos.shape[1] == 3
            my_gpos = <double*>gpos.data

        if vtens is None:
            my_vtens = NULL
        else:
            assert vtens.flags['C_CONTIGUOUS']
            assert vtens.shape[0] == 3
            assert vtens.shape[1] == 3
            my_vtens = <double*>vtens.data

        return pair_pot.pair_pot_compute(
            center_index, <nlists.nlist_row_type*>nlist.data, len(nlist),
            <pair_pot.scaling_row_type*>scaling.data, len(scaling),
            self._c_pair_pot, my_gpos, my_vtens
        )


cdef class PairPotLJ(PairPot):
    cdef np.ndarray _c_sigmas
    cdef np.ndarray _c_epsilons

    def __cinit__(self, np.ndarray[double, ndim=1] sigmas,
                  np.ndarray[double, ndim=1] epsilons, double cutoff, bint smooth):
        assert sigmas.flags['C_CONTIGUOUS']
        assert epsilons.flags['C_CONTIGUOUS']
        assert sigmas.shape[0] == epsilons.shape[0]
        pair_pot.pair_pot_set_cutoff(self._c_pair_pot, cutoff)
        pair_pot.pair_pot_set_smooth(self._c_pair_pot, smooth)
        pair_pot.pair_data_lj_init(self._c_pair_pot, <double*>sigmas.data, <double*>epsilons.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_sigmas = sigmas
        self._c_epsilons = epsilons

    def get_sigmas(self):
        return self._c_sigmas.view()

    sigmas = property(get_sigmas)

    def get_epsilons(self):
        return self._c_epsilons.view()

    epsilons = property(get_epsilons)


cdef class PairPotEI(PairPot):
    cdef np.ndarray charges

    def __cinit__(self, np.ndarray[double, ndim=1] charges, double alpha, double cutoff):
        assert charges.flags['C_CONTIGUOUS']
        pair_pot.pair_pot_set_cutoff(self._c_pair_pot, cutoff)
        pair_pot.pair_data_ei_init(self._c_pair_pot, <double*>charges.data, alpha)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self.charges = charges


#
# Ewald summation stuff
#


def compute_ewald_reci(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       Cell unitcell, double alpha, np.ndarray[long, ndim=1] gmax,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=1] work,
                       np.ndarray[double, ndim=2] vtens):
    cdef double *my_gpos
    cdef double *my_work
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert unitcell.nvec == 3
    assert alpha > 0
    assert gmax.flags['C_CONTIGUOUS']
    assert gmax.shape[0] == 3

    if gpos is None:
        my_gpos = NULL
        my_work = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        assert gpos.shape[0] == pos.shape[0]
        assert work.flags['C_CONTIGUOUS']
        assert gpos.shape[0]*2 == work.shape[0]
        my_gpos = <double*>gpos.data
        my_work = <double*>work.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    return ewald.compute_ewald_reci(<double*>pos.data, len(pos),
                                    <double*>charges.data,
                                    unitcell._c_cell, alpha,
                                    <long*>gmax.data, my_gpos, my_work,
                                    my_vtens)


def compute_ewald_corr(np.ndarray[double, ndim=2] pos,
                       long center_index,
                       np.ndarray[double, ndim=1] charges,
                       Cell unitcell, double alpha,
                       np.ndarray[pair_pot.scaling_row_type, ndim=1] scaling,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=2] vtens):
    cdef double *my_gpos
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert alpha > 0
    assert scaling.flags['C_CONTIGUOUS']

    if gpos is None:
        my_gpos = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        assert gpos.shape[0] == pos.shape[0]
        my_gpos = <double*>gpos.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    return ewald.compute_ewald_corr(<double*>pos.data, center_index,
                                    <double*>charges.data, unitcell._c_cell,
                                    alpha,
                                    <pair_pot.scaling_row_type*>scaling.data,
                                    len(scaling), my_gpos, my_vtens)


#
# Delta list
#


def dlist_forward(np.ndarray[double, ndim=2] pos,
                  Cell unitcell,
                  np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert deltas.flags['C_CONTIGUOUS']
    dlist.dlist_forward(<double*>pos.data, unitcell._c_cell,
                        <dlist.dlist_row_type*>deltas.data, ndelta)

def dlist_back(np.ndarray[double, ndim=2] gpos,
               np.ndarray[double, ndim=2] vtens,
               np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
    cdef double *my_gpos
    cdef double *my_vtens

    assert deltas.flags['C_CONTIGUOUS']
    if gpos is None and vtens is None:
        raise TypeError('Either gpos or vtens must be given.')

    if gpos is None:
        my_gpos = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        my_gpos = <double*>gpos.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    dlist.dlist_back(my_gpos, my_vtens,
                     <dlist.dlist_row_type*>deltas.data, ndelta)


#
# InternalCoordinate list
#


def iclist_forward(np.ndarray[dlist.dlist_row_type, ndim=1] deltas,
                   np.ndarray[iclist.iclist_row_type, ndim=1] ictab, long nic):
    assert deltas.flags['C_CONTIGUOUS']
    assert ictab.flags['C_CONTIGUOUS']
    iclist.iclist_forward(<dlist.dlist_row_type*>deltas.data,
                          <iclist.iclist_row_type*>ictab.data, nic)

def iclist_back(np.ndarray[dlist.dlist_row_type, ndim=1] deltas,
                np.ndarray[iclist.iclist_row_type, ndim=1] ictab, long nic):
    assert deltas.flags['C_CONTIGUOUS']
    assert ictab.flags['C_CONTIGUOUS']
    iclist.iclist_back(<dlist.dlist_row_type*>deltas.data,
                       <iclist.iclist_row_type*>ictab.data, nic)


#
# Valence list
#


def vlist_forward(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
                  np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    return vlist.vlist_forward(<iclist.iclist_row_type*>ictab.data,
                               <vlist.vlist_row_type*>vtab.data, nv)

def vlist_back(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
               np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    vlist.vlist_back(<iclist.iclist_row_type*>ictab.data,
                     <vlist.vlist_row_type*>vtab.data, nv)
