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


#ifndef YAFF_PAIRPOT_H
#define YAFF_PAIRPOT_H

#include "nlists.h"


typedef double (*pairfn_type)(void*, long, long, double, double*);

typedef struct {
  void *pairdata;
  pairfn_type pairfn;
  double cutoff;
} pairpot_type;

typedef struct {
  long i;
  double scale;
} scaling_row_type;


pairpot_type* pairpot_new(void);
void pairpot_free(pairpot_type *pairpot);
int pairpot_ready(pairpot_type *pairpot);
double pairpot_get_cutoff(pairpot_type *pairpot);
void pairpot_set_cutoff(pairpot_type *pairpot, double cutoff);

double pairpot_energy(long center_index, nlist_row_type *nlist,
                      long nlist_size, scaling_row_type *scaling,
                      long scaling_size, pairpot_type *pairpot);


typedef struct {
  double *sigma;
  double *epsilon;
} pairpot_lj_type;

void pairpot_lj_init(pairpot_type *pairpot, double *sigma, double *epsilon);
void pairpot_lj_free(pairpot_type *pairpot);
double pairpot_lj_eval(void *pairdata, long center_index, long other_index, double d, double *g);


#endif
