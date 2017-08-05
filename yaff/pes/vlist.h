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


#ifndef YAFF_VLIST_H
#define YAFF_VLIST_H

#include "iclist.h"

typedef struct {
  long kind;
  double par0, par1, par2, par3, par4, par5;
  long ic0, ic1;//, ic2;
  double energy;
} vlist_row_type;

double vlist_forward(iclist_row_type* ictab, vlist_row_type* vtab, long nv);
void vlist_back(iclist_row_type* ictab, vlist_row_type* vtab, long nv);

#endif
