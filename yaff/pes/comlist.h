// YAFF is yet another force-field code.
// Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
// --


#ifndef YAFF_COMLIST_H
#define YAFF_COMLIST_H

#include "dlist.h"

typedef struct {
  long i;    // atom or relative vector index.
  double w;  // group or atom weight
} comlist_row_type;

void comlist_forward(dlist_row_type* deltas, double *pos, double* compos, long* comsizes,
    comlist_row_type* comtab, long ncom);
void comlist_back(dlist_row_type* deltas, double *gpos, double* gcompos, long* comsizes,
    comlist_row_type* comtab, long ncom);

#endif
