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


#include "pair_pot.h"
#include <math.h>
#include <stdlib.h>


pair_pot_type* pair_pot_new(void) {
  pair_pot_type* result;
  result = malloc(sizeof(pair_pot_type));
  if (result != NULL) {
    (*result).pair_data = NULL;
    (*result).pair_fn = NULL;
    (*result).cutoff = 0.0;
  }
  return result;
}

void pair_pot_free(pair_pot_type *pair_pot) {
  free(pair_pot);
}

int pair_pot_ready(pair_pot_type *pair_pot) {
  return (*pair_pot).pair_data != NULL && (*pair_pot).pair_fn != NULL;
}

double pair_pot_get_cutoff(pair_pot_type *pair_pot) {
  return (*pair_pot).cutoff;
}

void pair_pot_set_cutoff(pair_pot_type *pair_pot, double cutoff) {
  (*pair_pot).cutoff = cutoff;
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


double pair_pot_energy(long center_index, nlist_row_type *nlist,
                       long nlist_size, scaling_row_type *scaling,
                       long scaling_size, pair_pot_type *pair_pot) {
  long i, other_index, scaling_counter;
  double s, energy, term;
  energy = 0.0;
  // Reset the counter for the scaling.
  scaling_counter = 0;
  // Compute the interactions.
  for (i=0; i<nlist_size; i++) {
    // Find the scale
    if (nlist[i].d < (*pair_pot).cutoff) {
      other_index = nlist[i].i;
      if ((nlist[i].r0 == 0) && (nlist[i].r1 == 0) && (nlist[i].r2 == 0)) {
        s = get_scaling(scaling, center_index, other_index, &scaling_counter, scaling_size);
      } else {
        s = 0.5;
      }
      // If the scale is non-zero, compute the contribution.
      if (s > 0.0) {
        // Call the potential function
        energy += s*(*pair_pot).pair_fn((*pair_pot).pair_data, center_index, other_index, nlist[i].d, NULL);
      }
    }
  }
  return energy;
}


void pair_data_lj_init(pair_pot_type *pair_pot, double *sigma, double *epsilon) {
  pair_data_lj_type *pair_data;
  pair_data = malloc(sizeof(pair_data_lj_type));
  (*pair_pot).pair_data = pair_data;
  (*pair_pot).pair_fn = pair_fn_lj;
  (*pair_data).sigma = sigma;
  (*pair_data).epsilon = epsilon;
}

void pair_data_lj_free(pair_pot_type *pair_pot) {
  free((*pair_pot).pair_data);
}

double pair_fn_lj(void *pair_data, long center_index, long other_index, double d, double *g) {
  double sigma, epsilon, x;
  sigma = 0.5*(
    (*(pair_data_lj_type*)pair_data).sigma[center_index]+
    (*(pair_data_lj_type*)pair_data).sigma[other_index]
  );
  epsilon = sqrt(
    (*(pair_data_lj_type*)pair_data).epsilon[center_index]*
    (*(pair_data_lj_type*)pair_data).epsilon[other_index]
  );
  x = sigma/d;
  x *= x;
  x *= x*x;
  //printf("C %3i %3i %10.5f %10.3e\n", center_index, other_index, d, 4.0*epsilon*(x*(x-1.0)));
  return 4.0*epsilon*(x*(x-1.0));
}
