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
#include "vlist.h"

typedef double (*v_forward_type)(vlist_row_type*, iclist_row_type*);

double forward_harmonic(vlist_row_type* term, iclist_row_type* ictab) {
  double x;
  x = ictab[(*term).ic0].value - (*term).par1;
  return 0.5*((*term).par0)*x*x;
}

double forward_polyfour(vlist_row_type* term, iclist_row_type* ictab) {
  double q = ictab[(*term).ic0].value;
  return (*term).par0*q + (*term).par1*q*q + (*term).par2*q*q*q + (*term).par3*q*q*q*q;
}

double forward_cross(vlist_row_type* term, iclist_row_type* ictab) {
  return (*term).par0*( ictab[(*term).ic0].value - (*term).par1 )*( ictab[(*term).ic1].value - (*term).par2 );
}

v_forward_type v_forward_fns[3] = {
  forward_harmonic, forward_polyfour, forward_cross
};

double vlist_forward(iclist_row_type* ictab, vlist_row_type* vtab, long nv) {
  long i;
  double energy;
  energy = 0.0;
  for (i=0; i<nv; i++) {
    energy += v_forward_fns[vtab[i].kind](vtab + i, ictab);
  }
  return energy;
}


typedef void (*v_back_type)(vlist_row_type*, iclist_row_type*);

void back_harmonic(vlist_row_type* term, iclist_row_type* ictab) {
  ictab[(*term).ic0].grad += ((*term).par0)*(ictab[(*term).ic0].value - (*term).par1);
}

void back_polyfour(vlist_row_type* term, iclist_row_type* ictab) {
  double q = ictab[(*term).ic0].value;
  ictab[(*term).ic0].grad += (*term).par0 + 2.0*(*term).par1*q + 3.0*(*term).par2*q*q + 4.0*(*term).par3*q*q*q;
}

void back_cross(vlist_row_type* term, iclist_row_type* ictab) {
  ictab[(*term).ic0].grad += (*term).par0*( ictab[(*term).ic1].value - (*term).par2 );
  ictab[(*term).ic1].grad += (*term).par0*( ictab[(*term).ic0].value - (*term).par1 );
}

v_back_type v_back_fns[3] = {
  back_harmonic, back_polyfour, back_cross
};

void vlist_back(iclist_row_type* ictab, vlist_row_type* vtab, long nv) {
  long i;
  for (i=0; i<nv; i++) {
    v_back_fns[vtab[i].kind](vtab + i, ictab);
  }
}
