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

double forward_fues(vlist_row_type* term, iclist_row_type* ictab) {
  double x;
  x = (*term).par1/ictab[(*term).ic0].value;
  return 0.5*(*term).par0*(*term).par1*(*term).par1*(1.0+x*(x-2.0));
}

double forward_cross(vlist_row_type* term, iclist_row_type* ictab) {
  return (*term).par0*( ictab[(*term).ic0].value - (*term).par1 )*( ictab[(*term).ic1].value - (*term).par2 );
}

double forward_cosine(vlist_row_type* term, iclist_row_type* ictab) {
  return 0.5*(*term).par1*(1-cos(
    (*term).par0*(ictab[(*term).ic0].value - (*term).par2)
  ));
}

double forward_chebychev1(vlist_row_type* term, iclist_row_type* ictab) {
  return 0.5*(*term).par0*(1+(*term).par1*ictab[(*term).ic0].value);
}

double forward_chebychev2(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  return 0.5*(*term).par0*(1+(*term).par1*(2*c*c-1));
}

double forward_chebychev3(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  return 0.5*(*term).par0*(1+(*term).par1*c*(4*c*c-3));
}

double forward_chebychev4(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  c = c*c;
  return 0.5*(*term).par0*(1+(*term).par1*(8*c*c-8*c+1));
}

double forward_chebychev6(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  c = c*c;
  return 0.5*(*term).par0*(1+(*term).par1*(32*c*c*c-48*c*c+18*c-1));
}

double forward_polysix(vlist_row_type* term, iclist_row_type* ictab) {
  double q = ictab[(*term).ic0].value;
  return (*term).par0*q + (*term).par1*q*q + (*term).par2*q*q*q + (*term).par3*q*q*q*q + (*term).par4*q*q*q*q*q + (*term).par5*q*q*q*q*q*q;
}

double forward_mm3quartic(vlist_row_type* term, iclist_row_type* ictab) {
  //the unit of the number 2.55 in the original MM3 paper is 1/angstrom. In yaff
  //we use atomic units as internal coordinates, hence a conversion of
  //1/angstrom to 1/bohr is required.
  //Finally, in the original MM3 paper, there is a typo in the quartic term, it
  //should be 2.55^2 instead of 2.55.
  double x = ictab[(*term).ic0].value - (*term).par1;
  double x2 = x*x;
  return 0.5*((*term).par0)*x2*(1.0-1.349402*x+1.062183*x2);
}

double forward_mm3bend(vlist_row_type* term, iclist_row_type* ictab) {
  //the unit of the coefficients in the sixth order expansion in the original
  //MM3 paper is 1/deg for the fourth order term, 1/deg^2 for the fifth order
  //term and so on. In yaff we use atomic units as internal coordinates, hence a
  //conversion of 1/deg to 1/rad is required.
  //Finally, in the original MM3 paper, there is a typo in the sixth order term,
  //it should be 2.2e-8 instead of 9e-10.
  double x = ictab[(*term).ic0].value - (*term).par1;
  double x2 = x*x;
  return 0.5*((*term).par0)*x2*(1.0-0.802141*x+0.183837*x2-0.131664*x2*x+0.237090*x2*x2);
}

double forward_bonddoublewell(vlist_row_type* term, iclist_row_type* ictab) {
  double K, temp;
  double x, y;
  temp = ((*term).par1-(*term).par2)*((*term).par1-(*term).par2);
  temp *= temp;
  K = (*term).par0/temp;
  x = ictab[(*term).ic0].value - (*term).par1;
  y = ictab[(*term).ic0].value - (*term).par2;
  y *= y;
  return 0.5*K*x*x*y*y;
}

double forward_morse(vlist_row_type* term, iclist_row_type* ictab) {
  double a;
  a = (*term).par1*(ictab[(*term).ic0].value-(*term).par2);
  return (*term).par0*(exp(-2.0*a)-2.0*exp(-a));
}

v_forward_type v_forward_fns[15] = {
  forward_harmonic, forward_polyfour, forward_fues, forward_cross,
  forward_cosine, forward_chebychev1, forward_chebychev2, forward_chebychev3,
  forward_chebychev4, forward_chebychev6, forward_polysix,
  forward_mm3quartic, forward_mm3bend, forward_bonddoublewell,
  forward_morse,
};

double vlist_forward(iclist_row_type* ictab, vlist_row_type* vtab, long nv) {
  long i;
  double energy;
  energy = 0.0;
  for (i=0; i<nv; i++) {
    vtab[i].energy = v_forward_fns[vtab[i].kind](vtab + i, ictab);
    energy += vtab[i].energy;
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

void back_fues(vlist_row_type* term, iclist_row_type* ictab) {
  double x = (*term).par1/ictab[(*term).ic0].value;
  ictab[(*term).ic0].grad += (*term).par0*(*term).par1*(x*x-x*x*x);
}

void back_cross(vlist_row_type* term, iclist_row_type* ictab) {
  ictab[(*term).ic0].grad += (*term).par0*( ictab[(*term).ic1].value - (*term).par2 );
  ictab[(*term).ic1].grad += (*term).par0*( ictab[(*term).ic0].value - (*term).par1 );
}

void back_cosine(vlist_row_type* term, iclist_row_type* ictab) {
  ictab[(*term).ic0].grad += 0.5*(*term).par1*(*term).par0*sin(
    (*term).par0*(ictab[(*term).ic0].value - (*term).par2)
  );
}

void back_chebychev1(vlist_row_type* term, iclist_row_type* ictab) {
  ictab[(*term).ic0].grad += 0.5*(*term).par0*(*term).par1;
}

void back_chebychev2(vlist_row_type* term, iclist_row_type* ictab) {
  ictab[(*term).ic0].grad += (*term).par1*2.0*(*term).par0*ictab[(*term).ic0].value;
}

void back_chebychev3(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  ictab[(*term).ic0].grad += (*term).par1*1.5*(*term).par0*(4*c*c-1);
}

void back_chebychev4(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  ictab[(*term).ic0].grad += (*term).par1*8*(*term).par0*c*(2*c*c-1);
}

void back_chebychev6(vlist_row_type* term, iclist_row_type* ictab) {
  double c;
  c = ictab[(*term).ic0].value;
  ictab[(*term).ic0].grad += (*term).par1*6*(*term).par0*c*(16*c*c*c*c-16*c*c+3);
}

void back_polysix(vlist_row_type* term, iclist_row_type* ictab) {
  double q = ictab[(*term).ic0].value;
  ictab[(*term).ic0].grad += (*term).par0 + 2.0*(*term).par1*q + 3.0*(*term).par2*q*q + 4.0*(*term).par3*q*q*q + 5.0*(*term).par4*q*q*q*q + 6.0*(*term).par5*q*q*q*q*q;
}

void back_mm3quartic(vlist_row_type* term, iclist_row_type* ictab) {
  //see comments in forward_mm3quartic
  double q = (ictab[(*term).ic0].value - (*term).par1);
  ictab[(*term).ic0].grad += ((*term).par0)*(q-2.024103*q*q+2.124366*q*q*q);
}

void back_mm3bend(vlist_row_type* term, iclist_row_type* ictab) {
  //see comments in forward_mm3bend
  double q = (ictab[(*term).ic0].value - (*term).par1);
  double q2 = q*q;
  ictab[(*term).ic0].grad += ((*term).par0)*(q-1.203211*q2+0.367674*q2*q-0.329159*q2*q2+0.711270*q2*q2*q);
}

void back_bonddoublewell(vlist_row_type* term, iclist_row_type* ictab) {
  double K, temp;
  double x, y, z;
  temp = ((*term).par1-(*term).par2)*((*term).par1-(*term).par2);
  temp *= temp;
  K = (*term).par0/(temp);
  x = ictab[(*term).ic0].value - (*term).par1;
  y = ictab[(*term).ic0].value - (*term).par2;
  y *= y;
  z = ictab[(*term).ic0].value - (*term).par2;
  ictab[(*term).ic0].grad += 0.5*K*(2*x*y*y+4*x*x*y*z);
}

void back_morse(vlist_row_type* term, iclist_row_type* ictab) {
  double a;
  a = (*term).par1*(ictab[(*term).ic0].value-(*term).par2);
  ictab[(*term).ic0].grad += -2.0*(*term).par1*(*term).par0*(exp(-2.0*a)-exp(-a));
}

v_back_type v_back_fns[15] = {
  back_harmonic, back_polyfour, back_fues, back_cross, back_cosine,
  back_chebychev1, back_chebychev2, back_chebychev3, back_chebychev4,
  back_chebychev6, back_polysix, back_mm3quartic,
  back_mm3bend, back_bonddoublewell, back_morse
};

void vlist_back(iclist_row_type* ictab, vlist_row_type* vtab, long nv) {
  long i;
  for (i=0; i<nv; i++) {
    v_back_fns[vtab[i].kind](vtab + i, ictab);
  }
}

typedef void (*v_hessian_type)(vlist_row_type*, iclist_row_type*, long nic, double* hessian);

void hessian_harmonic(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  hessian[index] += (*term).par0;
}

void hessian_polyfour(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double q = ictab[(*term).ic0].value;
  hessian[index] += 2.0*(*term).par1 + 6.0*(*term).par2*q + 12.0*(*term).par3*q*q;
}

void hessian_fues(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double x = (*term).par1/ictab[(*term).ic0].value;
  hessian[index] += (*term).par0*x*x*x*(3.0*x-2.0);
}

void hessian_cross(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*nic+(*term).ic1;
  hessian[index] += (*term).par0;
  index = (*term).ic1*nic+(*term).ic0;
  hessian[index] += (*term).par0;
}

void hessian_cosine(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  hessian[index] += 0.5*(*term).par1*(*term).par0*(*term).par0*cos(
    (*term).par0*(ictab[(*term).ic0].value - (*term).par2)
  );
}

void hessian_chebychev1(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  //Nothing to do here...
}

void hessian_chebychev2(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  hessian[index] += (*term).par1*2.0*(*term).par0;
}

void hessian_chebychev3(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double c;
  c = ictab[(*term).ic0].value;
  hessian[index] += (*term).par1*12*(*term).par0*c;
}

void hessian_chebychev4(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double c;
  c = ictab[(*term).ic0].value;
  hessian[index] += (*term).par1*8*(*term).par0*(6*c*c-1);
}

void hessian_chebychev6(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double c;
  c = ictab[(*term).ic0].value;
  hessian[index] += (*term).par1*6*(*term).par0*(80*c*c*c*c-48*c*c+3);
}

void hessian_polysix(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double q = ictab[(*term).ic0].value;
  hessian[index] += 2.0*(*term).par1 + 6.0*(*term).par2*q + 12.0*(*term).par3*q*q + 20.0*(*term).par4*q*q*q + 30.0*(*term).par5*q*q*q*q;
}

void hessian_morse(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  long index = (*term).ic0*(nic+1);
  double a;
  a = (*term).par1*(ictab[(*term).ic0].value-(*term).par2);
  hessian[index] += 2.0*(*term).par1*(*term).par0*(*term).par0*(2.0*exp(-2.0*a)-exp(-a));
}

void hessian_nan(vlist_row_type* term, iclist_row_type* ictab, long nic, double* hessian) {
  // Stub function for when people code new terms without implementing the hessian!
  long index = (*term).ic0*(nic+1);
  hessian[index] += NAN;
}

v_hessian_type v_hessian_fns[15] = {
  hessian_harmonic, hessian_polyfour, hessian_fues, hessian_cross, hessian_cosine,
  hessian_chebychev1, hessian_chebychev2, hessian_chebychev3, hessian_chebychev4,
  hessian_chebychev6, hessian_polysix, hessian_nan, hessian_nan, hessian_nan, hessian_morse
};

void vlist_hessian(iclist_row_type* ictab, vlist_row_type* vtab, long nv, long nic, double* hessian) {
  long i;
  for (i=0; i<nv; i++) {
    v_hessian_fns[vtab[i].kind](vtab + i, ictab, nic, hessian);
  }
}
