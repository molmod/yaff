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
cimport nlists
cimport pair_pot
cimport ewald
cimport dlist
cimport iclist
cimport vlist


__all__ = [
    'nlist_status_init', 'nlist_update', 'nlist_status_finish',
    'PairPot', 'PairPotLJ', 'PairPotEI', 'compute_ewald_reci',
    'compute_ewald_corr', 'dlist_forward', 'iclist_forward', 'vlist_forward',
]

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


def nlist_update(np.ndarray[double, ndim=2] pos, center_index, cutoff,
                 np.ndarray[long, ndim=1] rmax,
                 np.ndarray[double, ndim=2] rvecs,
                 np.ndarray[double, ndim=2] gvecs,
                 np.ndarray[long, ndim=1] nlist_status,
                 np.ndarray[nlists.nlist_row_type, ndim=1] nlist):
    assert pos.shape[1] == 3
    assert pos.flags['C_CONTIGUOUS']
    assert rmax.shape[0] <= 3
    assert rmax.flags['C_CONTIGUOUS']
    assert rvecs.shape[0] <= 3
    assert rvecs.shape[1] == 3
    assert rvecs.flags['C_CONTIGUOUS']
    assert gvecs.shape[0] <= 3
    assert gvecs.shape[1] == 3
    assert gvecs.flags['C_CONTIGUOUS']
    assert nlist_status.shape[0] == 5
    assert nlist_status.flags['C_CONTIGUOUS']
    assert nlist.flags['C_CONTIGUOUS']
    assert rmax.shape[0] == rvecs.shape[0]
    assert rmax.shape[0] == gvecs.shape[0]
    return nlists.nlist_update_low(
        <double*>pos.data, center_index, cutoff, <long*>rmax.data,
        <double*>rvecs.data, <double*>gvecs.data, <long*>nlist_status.data,
        <nlists.nlist_row_type*>nlist.data, len(pos), len(nlist), rvecs.shape[0]
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

    def compute(self, long center_index,
                np.ndarray[nlists.nlist_row_type, ndim=1] nlist,
                np.ndarray[pair_pot.scaling_row_type, ndim=1] scaling,
                np.ndarray[double, ndim=2] gradient):
        assert pair_pot.pair_pot_ready(self._c_pair_pot)
        assert nlist.flags['C_CONTIGUOUS']
        assert scaling.flags['C_CONTIGUOUS']
        if gradient is None:
            return pair_pot.pair_pot_energy_gradient(
                center_index, <nlists.nlist_row_type*>nlist.data, len(nlist),
                <pair_pot.scaling_row_type*>scaling.data, len(scaling),
                self._c_pair_pot, NULL
            )
        else:
            assert gradient.flags['C_CONTIGUOUS']
            assert gradient.shape[1] == 3
            return pair_pot.pair_pot_energy_gradient(
                center_index, <nlists.nlist_row_type*>nlist.data, len(nlist),
                <pair_pot.scaling_row_type*>scaling.data, len(scaling),
                self._c_pair_pot, <double*>gradient.data
            )


cdef class PairPotLJ(PairPot):
    def __cinit__(self, np.ndarray[double, ndim=1] sigmas,
                  np.ndarray[double, ndim=1] epsilons, double cutoff):
        assert sigmas.flags['C_CONTIGUOUS']
        assert epsilons.flags['C_CONTIGUOUS']
        assert sigmas.shape[0] == epsilons.shape[0]
        pair_pot.pair_pot_set_cutoff(self._c_pair_pot, cutoff)
        pair_pot.pair_data_lj_init(self._c_pair_pot, <double*>sigmas.data, <double*>epsilons.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()


cdef class PairPotEI(PairPot):
    def __cinit__(self, np.ndarray[double, ndim=1] charges, double alpha, double cutoff):
        assert charges.flags['C_CONTIGUOUS']
        pair_pot.pair_pot_set_cutoff(self._c_pair_pot, cutoff)
        pair_pot.pair_data_ei_init(self._c_pair_pot, <double*>charges.data, alpha)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()


#
# Ewald summation stuff
#


def compute_ewald_reci(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       np.ndarray[double, ndim=2] gvecs, double volume,
                       double alpha, np.ndarray[long, ndim=1] gmax,
                       np.ndarray[double, ndim=2] gradient,
                       np.ndarray[double, ndim=1] work):
    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert gvecs.flags['C_CONTIGUOUS']
    assert gvecs.shape[0] == 3
    assert gvecs.shape[1] == 3
    assert volume > 0
    assert alpha > 0
    assert gmax.flags['C_CONTIGUOUS']
    assert gmax.shape[0] == 3
    if gradient is None:
        return ewald.compute_ewald_reci(<double*>pos.data, len(pos),
                                        <double*>charges.data, <double*>gvecs.data,
                                        volume, alpha, <long*>gmax.data,
                                        NULL, NULL)
    else:
        assert gradient.flags['C_CONTIGUOUS']
        assert gradient.shape[1] == 3
        assert gradient.shape[0] == pos.shape[0]
        assert work.flags['C_CONTIGUOUS']
        assert gradient.shape[0]*2 == work.shape[0]
        return ewald.compute_ewald_reci(<double*>pos.data, len(pos),
                                        <double*>charges.data, <double*>gvecs.data,
                                        volume, alpha, <long*>gmax.data,
                                        <double*>gradient.data, <double*>work.data)


def compute_ewald_corr(np.ndarray[double, ndim=2] pos,
                       long center_index,
                       np.ndarray[double, ndim=1] charges,
                       np.ndarray[double, ndim=2] rvecs,
                       np.ndarray[double, ndim=2] gvecs, double alpha,
                       np.ndarray[pair_pot.scaling_row_type, ndim=1] scaling,
                       np.ndarray[double, ndim=2] gradient):
    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert rvecs.flags['C_CONTIGUOUS']
    assert rvecs.shape[0] == 3
    assert rvecs.shape[1] == 3
    assert gvecs.flags['C_CONTIGUOUS']
    assert gvecs.shape[0] == 3
    assert gvecs.shape[1] == 3
    assert alpha > 0
    assert scaling.flags['C_CONTIGUOUS']
    if gradient is None:
        return ewald.compute_ewald_corr(<double*>pos.data, center_index,
                                        <double*>charges.data, <double*>rvecs.data,
                                        <double*>gvecs.data, alpha,
                                        <pair_pot.scaling_row_type*>scaling.data,
                                        len(scaling), NULL)
    else:
        assert gradient.flags['C_CONTIGUOUS']
        assert gradient.shape[1] == 3
        assert gradient.shape[0] == pos.shape[0]
        return ewald.compute_ewald_corr(<double*>pos.data, center_index,
                                        <double*>charges.data, <double*>rvecs.data,
                                        <double*>gvecs.data, alpha,
                                        <pair_pot.scaling_row_type*>scaling.data,
                                        len(scaling), <double*>gradient.data)


#
# Delta list
#

def dlist_forward(np.ndarray[double, ndim=2] pos,
                  np.ndarray[double, ndim=2] rvecs,
                  np.ndarray[double, ndim=2] gvecs,
                  np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert rvecs.flags['C_CONTIGUOUS']
    assert rvecs.shape[1] == 3
    assert gvecs.flags['C_CONTIGUOUS']
    assert gvecs.shape[0] == rvecs.shape[0]
    assert gvecs.shape[1] == 3
    assert deltas.flags['C_CONTIGUOUS']
    dlist.dlist_forward(<double*>pos.data, <double*>rvecs.data,
                        <double*>gvecs.data, len(rvecs),
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


#
# Valence list
#

def vlist_forward(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
                  np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    return vlist.vlist_forward(<iclist.iclist_row_type*>ictab.data,
                               <vlist.vlist_row_type*>vtab.data, nv)
