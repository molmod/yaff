// YAFF is yet another force-field code
// Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
// for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
// reserved unless otherwise stated.
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
// --


#ifndef YAFF_CELL_H
#define YAFF_CELL_H

typedef struct {
  double rvecs[9], gvecs[9];
  double rspacings[3], gspacings[3];
  double volume;
  int nvec;
} cell_type;

cell_type* cell_new(void);
void cell_free(cell_type* cell);
void cell_update(cell_type* cell, double *rvecs, double *gvecs, int nvec);

int cell_get_nvec(cell_type* cell);
double cell_get_volume(cell_type* cell);
void cell_copy_rvecs(cell_type* cell, double *rvecs, int full);
void cell_copy_gvecs(cell_type* cell, double *gvecs, int full);
void cell_copy_rspacings(cell_type* cell, double *rspacings, int full);
void cell_copy_gspacings(cell_type* cell, double *gspacings, int full);

#endif
