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


cdef extern from "cell.h":
    ctypedef struct cell_type:
        pass

    cell_type* cell_new()
    void cell_free(cell_type* cell)
    void cell_update(cell_type* cell, double *rvecs, double *gvecs, int nvec)

    void cell_mic(double *delta, cell_type* cell)
    void cell_to_center(double *car, cell_type* cell, long *center)
    void cell_add_vec(double *delta, cell_type* cell, long* r)

    bint is_invalid_exclude(long* exclude, long natom0, long natom1, long nexclude, bint intra)
    void cell_compute_distances1(cell_type* cell, double* pos, double* output, long natom, long* pairs, long npair, bint do_include, long nimage)
    void cell_compute_distances2(cell_type* cell, double* pos0, double* pos1, double* output, long natom0, long natom1, long* pairs, long npair, bint do_include, long nimage)


    int cell_get_nvec(cell_type* cell)
    double cell_get_volume(cell_type* cell)
    void cell_copy_rvecs(cell_type* cell, double *rvecs, bint full)
    void cell_copy_gvecs(cell_type* cell, double *gvecs, bint full)
    void cell_copy_rspacings(cell_type* cell, double *rspacings, bint full)
    void cell_copy_gspacings(cell_type* cell, double *gspacings, bint full)
