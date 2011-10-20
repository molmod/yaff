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


cimport cell

cdef extern from "nlists.h":
    ctypedef struct nlist_row_type:
        long i
        double d
        double dx, dy, dz
        long r0, r1, r2

    bint nlist_update_low(double *pos, long center_index, double rcut,
                          long *rmax, cell.cell_type* cell, long
                          *nlist_status, nlist_row_type *nlist, long pos_size,
                          long nlist_size)

    bint nlist_inc_r(cell.cell_type *unitcell, long *r, long *rmax)
