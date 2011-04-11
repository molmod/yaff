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


#include "pairpot.h"
#include <math.h>
#include <stdlib.h>


pairpot_type* pairpot_new(void) {
  pairpot_type* result;
  result = malloc(sizeof(pairpot_type));
  if (result != NULL) {
    (*result).pairdata = NULL;
    (*result).pairfn = NULL;
    (*result).cutoff = 0.0;
  }
  return result;
}

void pairpot_free(pairpot_type *pairpot) {
  free(pairpot);
}

int pairpot_ready(pairpot_type *pairpot) {
  return (*pairpot).pairdata != NULL && (*pairpot).pairfn != NULL;
}

double pairpot_get_cutoff(pairpot_type *pairpot) {
  return (*pairpot).cutoff;
}

void pairpot_set_cutoff(pairpot_type *pairpot, double cutoff) {
  (*pairpot).cutoff = cutoff;
}


double get_scaling(scaling_row_type *scaling, long center_index, long other_index, long *counter, long size) {
  if (other_index==center_index) return 0.0;
  if (*counter >= size) return 1.0;
  while (scaling[*counter].i < other_index) {
    (*counter)++;
    if (*counter >= size) return 1.0;
  }
  if (scaling[*counter].i == other_index) {
    return scaling[*counter].scale;
  }
  return 1.0;
}


double pairpot_energy(long center_index, nlist_row_type *nlist,
                      long nlist_size, scaling_row_type *scaling,
                      long scaling_size, pairpot_type *pairpot) {
  long i, other_index, scaling_counter;
  double s, energy, term;
  energy = 0.0;
  // Reset the counter for the scaling.
  scaling_counter = 0;
  // Compute the interactions.
  for (i=0; i<nlist_size; i++) {
    // Find the scale
    if (nlist[i].d < (*pairpot).cutoff) {
      other_index = nlist[i].i;
      if ((nlist[i].r0 == 0) && (nlist[i].r1 == 0) && (nlist[i].r2 == 0)) {
        s = get_scaling(scaling, center_index, other_index, &scaling_counter, scaling_size);
      } else {
        s = 0.5;
      }
      // If the scale is non-zero, compute the contribution.
      if (s > 0.0) {
        // Call the potential function
        energy += s*(*pairpot).pairfn((*pairpot).pairdata, center_index, other_index, nlist[i].d, NULL);
      }
    }
  }
  return energy;
}


void pairpot_lj_init(pairpot_type *pairpot, double *sigma, double *epsilon) {
  pairpot_lj_type *pairdata;
  pairdata = malloc(sizeof(pairpot_lj_type));
  (*pairpot).pairdata = pairdata;
  (*pairpot).pairfn = pairpot_lj_eval;
  (*pairdata).sigma = sigma;
  (*pairdata).epsilon = epsilon;
}

void pairpot_lj_free(pairpot_type *pairpot) {
  free((*pairpot).pairdata);
}

double pairpot_lj_eval(void *pairdata, long center_index, long other_index, double d, double *g) {
  double sigma, epsilon, x;
  sigma = 0.5*(
    (*(pairpot_lj_type*)pairdata).sigma[center_index]+
    (*(pairpot_lj_type*)pairdata).sigma[other_index]
  );
  epsilon = sqrt(
    (*(pairpot_lj_type*)pairdata).epsilon[center_index]*
    (*(pairpot_lj_type*)pairdata).epsilon[other_index]
  );
  x = sigma/d;
  x *= x;
  x *= x*x;
  //printf("C %3i %3i %10.5f %10.3e\n", center_index, other_index, d, 4.0*epsilon*(x*(x-1.0)));
  return 4.0*epsilon*(x*(x-1.0));
}
