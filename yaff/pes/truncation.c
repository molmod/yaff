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


#include "truncation.h"
#include <math.h>
#include <stdlib.h>


double hammer(double d, double rcut, double tau, double *g) {
  double result, x;
  if (d < rcut) {
    x = d - rcut;
    result = exp(tau/x);
    if (g != NULL) *g = -result*tau/x/x;
  } else {
    result = 0.0;
    if (g != NULL) *g = 0.0;
  }
  return result;
}

trunc_scheme_type* hammer_new(double tau) {
  trunc_scheme_type* result;
  result = malloc(sizeof(trunc_scheme_type));
  if (result != NULL) {
    (*result).trunc_fn = hammer;
    (*result).par = tau;
  }
  return result;
}

double hammer_get_tau(trunc_scheme_type *trunc_scheme) {
  return (*trunc_scheme).par;
}



double switch3(double d, double rcut, double width, double *g) {
  double result, x;
  if (d < rcut) {
    x = rcut - d;
    if (x > width) {
      result = 1.0;
      if (g != NULL) *g = 0.0;
    } else {
      x /= width;
      result = (3 - 2*x)*x*x;
      if (g != NULL) *g = -6*x*(1-x)/width;
    }
  } else {
    result = 0.0;
    if (g != NULL) *g = 0.0;
  }
  return result;
}

trunc_scheme_type* switch3_new(double width) {
  trunc_scheme_type* result;
  result = malloc(sizeof(trunc_scheme_type));
  if (result != NULL) {
    (*result).trunc_fn = switch3;
    (*result).par = width;
  }
  return result;
}

double switch3_get_width(trunc_scheme_type *trunc_scheme){
  return (*trunc_scheme).par;
}


double trunc_scheme_fn(trunc_scheme_type *trunc_scheme, double d, double rcut, double *g) {
  return (*trunc_scheme).trunc_fn(d, rcut, (*trunc_scheme).par, g);
}

void trunc_scheme_free(trunc_scheme_type *trunc_scheme) {
  free(trunc_scheme);
}
