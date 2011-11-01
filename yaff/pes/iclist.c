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

double forward_dihed_cos(iclist_row_type* ic, dlist_row_type* deltas) {
  long i;
  double *delta0, *delta1, *delta2;
  double a[3], b[3];
  double tmp0, tmp1, tmp2;
  delta0 = (double*)(deltas + (*ic).i0);
  delta1 = (double*)(deltas + (*ic).i1);
  delta2 = (double*)(deltas + (*ic).i2);
  tmp1 = sqrt(delta1[0]*delta1[0] + delta1[1]*delta1[1] + delta1[2]*delta1[2]);
  tmp0 = (delta0[0]*delta1[0] + delta0[1]*delta1[1] + delta0[2]*delta1[2])/tmp1;
  tmp2 = (delta1[0]*delta2[0] + delta1[1]*delta2[1] + delta1[2]*delta2[2])/tmp1;
  for (i=0; i<3; i++) {
    a[i] = delta0[i] - tmp0*delta1[i]/tmp1;
    b[i] = delta2[i] - tmp2*delta1[i]/tmp1;
  }
  tmp0 = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
  tmp2 = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  tmp1 = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  return (*ic).sign0*(*ic).sign2*tmp1/tmp0/tmp2;
}

double forward_dihed_angle(iclist_row_type* ic, dlist_row_type* deltas) {
  double c;
  c = forward_dihed_cos(ic, deltas);
  // Guard against round-off errors before taking the dot product.
  if (c > 1) {
    c = 1;
  } else if (c < -1) {
    c = -1;
  }
  // TODO: This is only correct for pure cosines of the dihedral angle. One
  //       should add a sign convention for the angle to model chiral stuff.
  return acos(c);
}

ic_forward_type ic_forward_fns[6] = {
  forward_bond, forward_bend_cos, forward_bend_angle, forward_dihed_cos, forward_dihed_angle, forward_bond
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

void back_dihed_cos(iclist_row_type* ic, dlist_row_type* deltas, double value, double grad) {
  long i;
  dlist_row_type *delta0, *delta1, *delta2;
  double a[3], b[3], dcos_da[3], dcos_db[3], da_ddel0[9], da_ddel1[9], db_ddel1[9];
  double dot0, dot2, n1, na, nb;

  delta0 = deltas + (*ic).i0;
  delta1 = deltas + (*ic).i1;
  delta2 = deltas + (*ic).i2;
  n1   = sqrt((*delta1).dx*(*delta1).dx + (*delta1).dy*(*delta1).dy + (*delta1).dz*(*delta1).dz);
  dot0 =      (*delta0).dx*(*delta1).dx + (*delta0).dy*(*delta1).dy + (*delta0).dz*(*delta1).dz;
  dot2 =      (*delta1).dx*(*delta2).dx + (*delta1).dy*(*delta2).dy + (*delta1).dz*(*delta2).dz;

  a[0] = ( (*delta0).dx - dot0*(*delta1).dx/(n1*n1) );
  a[1] = ( (*delta0).dy - dot0*(*delta1).dy/(n1*n1) );
  a[2] = ( (*delta0).dz - dot0*(*delta1).dz/(n1*n1) );
  b[0] = ( (*delta2).dx - dot2*(*delta1).dx/(n1*n1) );
  b[1] = ( (*delta2).dy - dot2*(*delta1).dy/(n1*n1) );
  b[2] = ( (*delta2).dz - dot2*(*delta1).dz/(n1*n1) );

  na = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
  nb = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);

  value *= (*ic).sign0*(*ic).sign2;
  grad *= (*ic).sign0*(*ic).sign2;

  for (i=0; i<3; i++) {
    dcos_da[i] = (b[i]/nb - value*a[i]/na)/na;
    dcos_db[i] = (a[i]/na - value*b[i]/nb)/nb;
  }

  da_ddel0[0] = 1 - (*delta1).dx*(*delta1).dx/(n1*n1);
  da_ddel0[1] =   - (*delta1).dx*(*delta1).dy/(n1*n1);
  da_ddel0[2] =   - (*delta1).dx*(*delta1).dz/(n1*n1);
  da_ddel0[3] =   - (*delta1).dy*(*delta1).dx/(n1*n1);
  da_ddel0[4] = 1 - (*delta1).dy*(*delta1).dy/(n1*n1);
  da_ddel0[5] =   - (*delta1).dy*(*delta1).dz/(n1*n1);
  da_ddel0[6] =   - (*delta1).dz*(*delta1).dx/(n1*n1);
  da_ddel0[7] =   - (*delta1).dz*(*delta1).dy/(n1*n1);
  da_ddel0[8] = 1 - (*delta1).dz*(*delta1).dz/(n1*n1);

  da_ddel1[0] = ( - dot0/(n1*n1) - (*delta0).dx*(*delta1).dx/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dx*(*delta1).dx );
  da_ddel1[1] = (                - (*delta0).dx*(*delta1).dy/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dx*(*delta1).dy );
  da_ddel1[2] = (                - (*delta0).dx*(*delta1).dz/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dx*(*delta1).dz );
  da_ddel1[3] = (                - (*delta0).dy*(*delta1).dx/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dy*(*delta1).dx );
  da_ddel1[4] = ( - dot0/(n1*n1) - (*delta0).dy*(*delta1).dy/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dy*(*delta1).dy );
  da_ddel1[5] = (                - (*delta0).dy*(*delta1).dz/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dy*(*delta1).dz );
  da_ddel1[6] = (                - (*delta0).dz*(*delta1).dx/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dz*(*delta1).dx );
  da_ddel1[7] = (                - (*delta0).dz*(*delta1).dy/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dz*(*delta1).dy );
  da_ddel1[8] = ( - dot0/(n1*n1) - (*delta0).dz*(*delta1).dz/(n1*n1) + 2*dot0/(n1*n1*n1*n1)*(*delta1).dz*(*delta1).dz );

  db_ddel1[0] = ( - dot2/(n1*n1) - (*delta2).dx*(*delta1).dx/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dx*(*delta1).dx );
  db_ddel1[1] = (                - (*delta2).dx*(*delta1).dy/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dx*(*delta1).dy );
  db_ddel1[2] = (                - (*delta2).dx*(*delta1).dz/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dx*(*delta1).dz );
  db_ddel1[3] = (                - (*delta2).dy*(*delta1).dx/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dy*(*delta1).dx );
  db_ddel1[4] = ( - dot2/(n1*n1) - (*delta2).dy*(*delta1).dy/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dy*(*delta1).dy );
  db_ddel1[5] = (                - (*delta2).dy*(*delta1).dz/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dy*(*delta1).dz );
  db_ddel1[6] = (                - (*delta2).dz*(*delta1).dx/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dz*(*delta1).dx );
  db_ddel1[7] = (                - (*delta2).dz*(*delta1).dy/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dz*(*delta1).dy );
  db_ddel1[8] = ( - dot2/(n1*n1) - (*delta2).dz*(*delta1).dz/(n1*n1) + 2*dot2/(n1*n1*n1*n1)*(*delta1).dz*(*delta1).dz );

  (*delta0).gx += grad*(  dcos_da[0]*da_ddel0[0] + dcos_da[1]*da_ddel0[3] + dcos_da[2]*da_ddel0[6]);
  (*delta0).gy += grad*(  dcos_da[0]*da_ddel0[1] + dcos_da[1]*da_ddel0[4] + dcos_da[2]*da_ddel0[7]);
  (*delta0).gz += grad*(  dcos_da[0]*da_ddel0[2] + dcos_da[1]*da_ddel0[5] + dcos_da[2]*da_ddel0[8]);
  (*delta1).gx += grad*(  dcos_da[0]*da_ddel1[0] + dcos_da[1]*da_ddel1[3] + dcos_da[2]*da_ddel1[6]
                        + dcos_db[0]*db_ddel1[0] + dcos_db[1]*db_ddel1[3] + dcos_db[2]*db_ddel1[6]);
  (*delta1).gy += grad*(  dcos_da[0]*da_ddel1[1] + dcos_da[1]*da_ddel1[4] + dcos_da[2]*da_ddel1[7]
                        + dcos_db[0]*db_ddel1[1] + dcos_db[1]*db_ddel1[4] + dcos_db[2]*db_ddel1[7]);
  (*delta1).gz += grad*(  dcos_da[0]*da_ddel1[2] + dcos_da[1]*da_ddel1[5] + dcos_da[2]*da_ddel1[8]
                        + dcos_db[0]*db_ddel1[2] + dcos_db[1]*db_ddel1[5] + dcos_db[2]*db_ddel1[8]);
  (*delta2).gx += grad*(  dcos_db[0]*da_ddel0[0] + dcos_db[1]*da_ddel0[3] + dcos_db[2]*da_ddel0[6]);
  (*delta2).gy += grad*(  dcos_db[0]*da_ddel0[1] + dcos_db[1]*da_ddel0[4] + dcos_db[2]*da_ddel0[7]);
  (*delta2).gz += grad*(  dcos_db[0]*da_ddel0[2] + dcos_db[1]*da_ddel0[5] + dcos_db[2]*da_ddel0[8]);
}

void back_dihed_angle(iclist_row_type* ic, dlist_row_type* deltas, double value, double grad) {
  back_dihed_cos(ic, deltas, cos(value), -grad/sin(value));
}

ic_back_type ic_back_fns[6] = {
  back_bond, back_bend_cos, back_bend_angle, back_dihed_cos, back_dihed_angle, back_bond
};

void iclist_back(dlist_row_type* deltas, iclist_row_type* ictab, long nic) {
  long i;
  for (i=0; i<nic; i++) {
    ic_back_fns[ictab[i].kind](ictab + i, deltas, ictab[i].value, ictab[i].grad);
  }
}
