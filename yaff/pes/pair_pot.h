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


#ifndef YAFF_PAIR_POT_H
#define YAFF_PAIR_POT_H

#include "nlist.h"
#include "truncation.h"
#include "slater.h"


typedef double (*pair_fn_type)(void*, long, long, double, double*, double*, double*);

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
double pair_fn_lj(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  double *sigma;
  double *epsilon;
  int *onlypauli;
} pair_data_mm3_type;

void pair_data_mm3_init(pair_pot_type *pair_pot, double *sigma, double *epsilon, int *onlypauli);
double pair_fn_mm3(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  double *r0;
  double *c6;
} pair_data_grimme_type;

void pair_data_grimme_init(pair_pot_type *pair_pot, double *r0, double *c6);
double pair_fn_grimme(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  long nffatype;
  long *ffatype_ids;
  double *amp_cross;
  double *b_cross;
} pair_data_exprep_type;

void pair_data_exprep_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *amp_cross, double *b_cross);
double pair_fn_exprep(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  long nffatype;
  long *ffatype_ids;
  double *amp_cross;
  double *b_cross;
} pair_data_qmdffrep_type;

void pair_data_qmdffrep_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *amp_cross, double *b_cross);
double pair_fn_qmdffrep(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  long nffatype;
  long *ffatype_ids;
  double *eps_cross;
  double *sig_cross;
} pair_data_ljcross_type;


void pair_data_ljcross_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *eps_cross, double *sig_cross);
double pair_fn_ljcross(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  long nffatype;
  long power;
  long *ffatype_ids;
  double *cn_cross;
  double *b_cross;
} pair_data_dampdisp_type;

void pair_data_dampdisp_init(pair_pot_type *pair_pot, long nffatype, long power, long* ffatype_ids, double *cn_cross, double *b_cross);
double pair_fn_dampdisp(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  long nffatype;
  long *ffatype_ids;
  double *c6_cross;
  double *c8_cross;
  double *R_cross;
  double c6_scale;
  double c8_scale;
  double bj_a;
  double bj_b;
} pair_data_disp68bjdamp_type;

void pair_data_disp68bjdamp_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *c6_cross, double *c8_cross, double *R_cross, double c6_scale, double c8_scale, double bj_a, double bj_b);
double pair_fn_disp68bjdamp(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);
double pair_data_disp68bjdamp_get_c6_scale(pair_pot_type *pair_pot);
double pair_data_disp68bjdamp_get_c8_scale(pair_pot_type *pair_pot);
double pair_data_disp68bjdamp_get_bj_a(pair_pot_type *pair_pot);
double pair_data_disp68bjdamp_get_bj_b(pair_pot_type *pair_pot);


typedef struct {
  double *charges;
  double alpha;
  double dielectric;
  double *radii;
} pair_data_ei_type;

void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha, double dielectric, double *radii);
double pair_fn_ei(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);
double pair_data_ei_get_alpha(pair_pot_type *pair_pot);
double pair_data_ei_get_dielectric(pair_pot_type *pair_pot);


typedef struct {
  double *charges;
  double *dipoles;
  double alpha;
  double *radii;
  double *radii2;
} pair_data_eidip_type;

void pair_data_eidip_init(pair_pot_type *pair_pot, double *charges, double *dipoles, double alpha, double *radii, double *radii2);
double pair_fn_eidip(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);
double pair_data_eidip_get_alpha(pair_pot_type *pair_pot);


typedef struct {
  double *Ns;
  double *Zs;
  double *widthss;
  double *Np;
  double *Zp;
  double *widthsp;
} pair_data_eislater1sp1spcorr_type;

void pair_data_eislater1sp1spcorr_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double *slater1s_Z, double *slater1p_widths, double *slater1p_N, double *slater1p_Z);
double pair_fn_eislater1sp1spcorr(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  double *N;
  double *Z;
  double *widths;
} pair_data_eislater1s1scorr_type;

void pair_data_eislater1s1scorr_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double *slater1s_Z);
double pair_fn_eislater1s1scorr(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);


typedef struct {
  double *N;
  double *widths;
  double ex_scale;
  double corr_a;
  double corr_b;
  double corr_c;
} pair_data_olpslater1s1s_type;

void pair_data_olpslater1s1s_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double ex_scale, double corr_a, double corr_b, double corr_c);
double pair_fn_olpslater1s1s(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);
double pair_data_olpslater1s1s_get_ex_scale(pair_pot_type *pair_pot);
double pair_data_olpslater1s1s_get_corr_a(pair_pot_type *pair_pot);
double pair_data_olpslater1s1s_get_corr_b(pair_pot_type *pair_pot);
double pair_data_olpslater1s1s_get_corr_c(pair_pot_type *pair_pot);


typedef struct {
  double *N;
  double *widths;
  double ct_scale;
  double width_power;
} pair_data_chargetransferslater1s1s_type;

void pair_data_chargetransferslater1s1s_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double ct_scale, double width_power);
double pair_fn_chargetransferslater1s1s(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart);
double pair_data_chargetransferslater1s1s_get_ct_scale(pair_pot_type *pair_pot);
double pair_data_chargetransferslater1s1s_get_width_power(pair_pot_type *pair_pot);
#endif
