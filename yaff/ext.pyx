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


cdef extern from "nlists.h":
    ctypedef struct nlist_row_type:
        np.long_t i
        np.float64_t d
        np.float64_t dx, dy, dz
        np.long_t r0, r1, r2

    int nlist_update_low(double *pos, long center_index, double cutoff,
                         long *rmax, double *rvecs, double *gvecs, long
                         *nlist_status, nlist_row_type *nlist, long pos_size,
                         long nlist_size, int nvec)


def nlist_status_init(rmax):
    # five integer status fields:
    # * r0
    # * r1
    # * r2
    # * other_index
    # * number of rows consumed
    return np.array([-rmax[0], -rmax[1], -rmax[2], 0, 0], int)


def nlist_update(np.ndarray[np.float64_t, ndim=2] pos, center_index, cutoff,
                 np.ndarray[np.long_t, ndim=1] rmax,
                 np.ndarray[np.float64_t, ndim=2] rvecs,
                 np.ndarray[np.float64_t, ndim=2] gvecs,
                 np.ndarray[np.long_t, ndim=1] nlist_status,
                 np.ndarray[nlist_row_type, ndim=1] nlist):
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
    return nlist_update_low(
        <double*>pos.data, center_index, cutoff, <long*>rmax.data,
        <double*>rvecs.data, <double*>gvecs.data, <long*>nlist_status.data,
        <nlist_row_type*>nlist.data, len(pos), len(nlist), rvecs.shape[0]
    )


def nlist_status_finish(nlist_status):
    return nlist_status[4]
