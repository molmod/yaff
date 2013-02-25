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


#ifndef YAFF_PAIR_POT_H
#define YAFF_PAIR_POT_H

#include "nlist.h"
#include "truncation.h"


typedef double (*pair_fn_type)(void*, long, long, double, double*);

typedef struct {
  void *pair_data;
  pair_fn_type pair_fn;
  double rcut;
  trunc_scheme_type *trunc_scheme;
} pair_pot_type;

typedef struct {
  long a, b;
  double scale;
  long nbond;
} scaling_row_type;


pair_pot_type* pair_pot_new(void);
void pair_pot_free(pair_pot_type *pair_pot);
int pair_pot_ready(pair_pot_type *pair_pot);
double pair_pot_get_rcut(pair_pot_type *pair_pot);
void pair_pot_set_rcut(pair_pot_type *pair_pot, double rcut);
void pair_pot_set_trunc_scheme(pair_pot_type *pair_pot, trunc_scheme_type *trunc_sceme);
void pair_data_free(pair_pot_type *pair_pot);

double pair_pot_compute(neigh_row_type *neighs,
                        long nneigh, scaling_row_type *scaling,
                        long scaling_size, pair_pot_type *pair_pot,
                        double *gpos, double* vtens);


typedef struct {
  double *sigma;
  double *epsilon;
} pair_data_lj_type;

void pair_data_lj_init(pair_pot_type *pair_pot, double *sigma, double *epsilon);
double pair_fn_lj(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  double *sigma;
  double *epsilon;
  int *onlypauli;
} pair_data_mm3_type;

void pair_data_mm3_init(pair_pot_type *pair_pot, double *sigma, double *epsilon, int *onlypauli);
double pair_fn_mm3(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  double *r0;
  double *c6;
} pair_data_grimme_type;

void pair_data_grimme_init(pair_pot_type *pair_pot, double *r0, double *c6);
double pair_fn_grimme(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  long nffatype;
  long *ffatype_ids;
  double *amp_cross;
  double *b_cross;
} pair_data_exprep_type;

void pair_data_exprep_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *amp_cross, double *b_cross);
double pair_fn_exprep(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  long nffatype;
  long *ffatype_ids;
  double *c6_cross;
  double *b_cross;
} pair_data_dampdisp_type;

void pair_data_dampdisp_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *c6_cross, double *b_cross);
double pair_fn_dampdisp(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  double *charges;
  double alpha;
} pair_data_ei_type;

void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha);
double pair_fn_ei(void *pair_data, long center_index, long other_index, double d, double *g);
double pair_data_ei_get_alpha(pair_pot_type *pair_pot);


#endif
