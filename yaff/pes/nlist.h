// YAFF is yet another force-field code
// Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
// Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
// (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
// stated.
//
// This file is part of YAFF.
//
// YAFF is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// YAFF is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>
//
//--


#ifndef YAFF_NLIST_H
#define YAFF_NLIST_H

#include "cell.h"

typedef struct {
    long a, b;
    double d;
    double dx, dy, dz;
    long r0, r1, r2;
} neigh_row_type;

int nlist_build_low(double *pos, double rcut, long *rmax, cell_type *unitcell,
                    long *nlist_status, neigh_row_type *neighs, long pos_size,
                    long nneigh);

void nlist_recompute_low(double *pos, double *pos_old, cell_type* unitcell,
                         neigh_row_type *neighs, long nneigh);

int nlist_inc_r(cell_type *unitcell, long *r, long *rmax);

#endif
