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


#include <math.h>
#include "iclist.h"

typedef double (*ic_forward_type)(iclist_row_type* ic, dlist_row_type* deltas);

double forward_bond(iclist_row_type* ic, dlist_row_type* deltas) {
  double *delta;
  delta = (double*)(deltas + (*ic).i0);
  return sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
}

double forward_bend_cos(iclist_row_type* ic, dlist_row_type* deltas) {
  double *delta0, *delta1;
  double d0, d1, dot;
  delta0 = (double*)(deltas + (*ic).i0);
  delta1 = (double*)(deltas + (*ic).i1);
  d0 = sqrt(delta0[0]*delta0[0] + delta0[1]*delta0[1] + delta0[2]*delta0[2]);
  d1 = sqrt(delta1[0]*delta1[0] + delta1[1]*delta1[1] + delta1[2]*delta1[2]);
  if ((d0 == 0) || (d1 == 0)) return 0.0;
  dot = delta0[0]*delta1[0] + delta0[1]*delta1[1] + delta0[2]*delta1[2];
  return (*ic).sign0*(*ic).sign1*dot/d0/d1;
}


double forward_bend_angle(iclist_row_type* ic, dlist_row_type* deltas) {
  double c;
  c = forward_bend_cos(ic, deltas);
  return acos(c);
}

ic_forward_type ic_forward_fns[3] = {
  forward_bond, forward_bend_cos, forward_bend_angle
};

void iclist_forward(dlist_row_type* deltas, iclist_row_type* ics, long nic) {
  long i;
  for (i=0; i<nic; i++) {
    ics[i].value = ic_forward_fns[ics[i].kind](ics + i, deltas);
  }
}
