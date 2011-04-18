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

typedef double (*ic_forward_type)(iclist_row_type*, dlist_row_type*);

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

void iclist_forward(dlist_row_type* deltas, iclist_row_type* ictab, long nic) {
  long i;
  for (i=0; i<nic; i++) {
    ictab[i].value = ic_forward_fns[ictab[i].kind](ictab + i, deltas);
    ictab[i].grad = 0.0;
  }
}


typedef void (*ic_back_type)(iclist_row_type*, dlist_row_type*, double, double);

void back_bond(iclist_row_type* ic, dlist_row_type* deltas, double value, double grad) {
  dlist_row_type *delta;
  double x;
  delta = deltas + (*ic).i0;
  x = grad/value;
  (*delta).gx += x*(*delta).dx;
  (*delta).gy += x*(*delta).dy;
  (*delta).gz += x*(*delta).dz;
}

void back_bend_cos(iclist_row_type* ic, dlist_row_type* deltas, double value, double grad) {
  dlist_row_type *delta0, *delta1;
  double e0[3], e1[3];
  double d0, d1, fac;
  delta0 = deltas + (*ic).i0;
  delta1 = deltas + (*ic).i1;
  d0 = sqrt((*delta0).dx*(*delta0).dx + (*delta0).dy*(*delta0).dy + (*delta0).dz*(*delta0).dz);
  d1 = sqrt((*delta1).dx*(*delta1).dx + (*delta1).dy*(*delta1).dy + (*delta1).dz*(*delta1).dz);
  e0[0] = (*delta0).dx/d0;
  e0[1] = (*delta0).dy/d0;
  e0[2] = (*delta0).dz/d0;
  e1[0] = (*delta1).dx/d1;
  e1[1] = (*delta1).dy/d1;
  e1[2] = (*delta1).dz/d1;
  fac = (*ic).sign0*(*ic).sign1;
  grad *= fac;
  value *= fac;
  fac = grad/d0;
  (*delta0).gx += fac*(e1[0] - value*e0[0]);
  (*delta0).gy += fac*(e1[1] - value*e0[1]);
  (*delta0).gz += fac*(e1[2] - value*e0[2]);
  fac = grad/d1;
  (*delta1).gx += fac*(e0[0] - value*e1[0]);
  (*delta1).gy += fac*(e0[1] - value*e1[1]);
  (*delta1).gz += fac*(e0[2] - value*e1[2]);
}

void back_bend_angle(iclist_row_type* ic, dlist_row_type* deltas, double value, double grad) {
  back_bend_cos(ic, deltas, cos(value), -grad/sin(value));
}

ic_back_type ic_back_fns[3] = {
  back_bond, back_bend_cos, back_bend_angle
};

void iclist_back(dlist_row_type* deltas, iclist_row_type* ictab, long nic) {
  long i;
  for (i=0; i<nic; i++) {
    ic_back_fns[ictab[i].kind](ictab + i, deltas, ictab[i].value, ictab[i].grad);
  }
}
