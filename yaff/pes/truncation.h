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


#ifndef YAFF_TRUNCATION_H
#define YAFF_TRUNCATION_H


typedef double (*trunc_fn_type)(double, double, double, double*);

typedef struct {
  trunc_fn_type trunc_fn;
  double par;
} trunc_scheme_type;

trunc_scheme_type* hammer_new(double tau);
double hammer_get_tau(trunc_scheme_type *trunc_scheme);

trunc_scheme_type* switch3_new(double width);
double switch3_get_width(trunc_scheme_type *trunc_scheme);

double trunc_scheme_fn(trunc_scheme_type *trunc_scheme, double d, double rcut, double *g);
void trunc_scheme_free(trunc_scheme_type *trunc_scheme);


#endif
