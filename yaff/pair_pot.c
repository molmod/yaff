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
#include <stdlib.h>
#include "constants.h"
#include "pair_pot.h"


pair_pot_type* pair_pot_new(void) {
  pair_pot_type* result;
  result = malloc(sizeof(pair_pot_type));
  if (result != NULL) {
    (*result).pair_data = NULL;
    (*result).pair_fn = NULL;
    (*result).cutoff = 0.0;
    (*result).smooth = 0;
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

int pair_pot_get_smooth(pair_pot_type *pair_pot) {
  return (*pair_pot).smooth;
}

void pair_pot_set_smooth(pair_pot_type *pair_pot, int smooth) {
  (*pair_pot).smooth = smooth;
}


double get_scaling(scaling_row_type *scaling, long center_index, long other_index, long *counter, long size) {
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


double hammer(double d, double c, double *g) {
  double result, x;
  if (d < c) {
    x = d - c;
    result = exp(1.0/x);
    if (g != NULL) *g = -result/x/x;
  } else {
    result = 0.0;
    if (g != NULL) *g = 0.0;
  }
  return result;
}


double pair_pot_compute(long center_index, nlist_row_type *nlist,
                        long nlist_size, scaling_row_type *scaling,
                        long scaling_size, pair_pot_type *pair_pot,
                        double *gpos, double* vtens) {
  long i, other_index, scaling_counter;
  double s, energy, v, vg, h, hg;
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
        if ((gpos==NULL) && (vtens==NULL)) {
          // Call the potential function without g argument.
          v = (*pair_pot).pair_fn((*pair_pot).pair_data, center_index, other_index, nlist[i].d, NULL);
          if ((*pair_pot).smooth) v *= hammer(nlist[i].d, (*pair_pot).cutoff, NULL);
        } else {
          // Call the potential function with vg argument.
          // vg is the derivative of the pair potential divided by the distance.
          v = (*pair_pot).pair_fn((*pair_pot).pair_data, center_index, other_index, nlist[i].d, &vg);
          if ((*pair_pot).smooth) {
            h = hammer(nlist[i].d, (*pair_pot).cutoff, &hg);
            vg = vg*h + v*hg/nlist[i].d;
            v *= h;
          }
          vg *= s;
          if (gpos!=NULL) {
            gpos[3*center_index  ] += nlist[i].dx*vg;
            gpos[3*center_index+1] += nlist[i].dy*vg;
            gpos[3*center_index+2] += nlist[i].dz*vg;
            gpos[3*other_index  ] -= nlist[i].dx*vg;
            gpos[3*other_index+1] -= nlist[i].dy*vg;
            gpos[3*other_index+2] -= nlist[i].dz*vg;
          }
          if (vtens!=NULL) {
            vtens[0] += nlist[i].dx*nlist[i].dx*vg;
            vtens[1] += nlist[i].dx*nlist[i].dy*vg;
            vtens[2] += nlist[i].dx*nlist[i].dz*vg;
            vtens[3] += nlist[i].dy*nlist[i].dx*vg;
            vtens[4] += nlist[i].dy*nlist[i].dy*vg;
            vtens[5] += nlist[i].dy*nlist[i].dz*vg;
            vtens[6] += nlist[i].dz*nlist[i].dx*vg;
            vtens[7] += nlist[i].dz*nlist[i].dy*vg;
            vtens[8] += nlist[i].dz*nlist[i].dz*vg;
          }
        }
        energy += s*v;
      }
    }
  }
  return energy;
}

void pair_data_free(pair_pot_type *pair_pot) {
  free((*pair_pot).pair_data);
  (*pair_pot).pair_data = NULL;
  (*pair_pot).pair_fn = NULL;
}



void pair_data_lj_init(pair_pot_type *pair_pot, double *sigma, double *epsilon) {
  pair_data_lj_type *pair_data;
  pair_data = malloc(sizeof(pair_data_lj_type));
  (*pair_pot).pair_data = pair_data;
  (*pair_pot).pair_fn = pair_fn_lj;
  (*pair_data).sigma = sigma;
  (*pair_data).epsilon = epsilon;
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
  if (g != NULL) {
    *g = 24.0*epsilon/d/d*x*(1.0-2.0*x);
  }
  return 4.0*epsilon*(x*(x-1.0));
}


void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha) {
  pair_data_ei_type *pair_data;
  pair_data = malloc(sizeof(pair_data_ei_type));
  (*pair_pot).pair_data = pair_data;
  (*pair_pot).pair_fn = pair_fn_ei;
  (*pair_data).charges = charges;
  (*pair_data).alpha = alpha;
}

double pair_fn_ei(void *pair_data, long center_index, long other_index, double d, double *g) {
  double pot, alpha, qprod, x;
  qprod = (
    (*(pair_data_ei_type*)pair_data).charges[center_index]*
    (*(pair_data_ei_type*)pair_data).charges[other_index]
  );
  alpha = (*(pair_data_ei_type*)pair_data).alpha;
  if (alpha > 0) {
    x = alpha*d;
    pot = erfc(x)/d;
  } else {
    pot = 1.0/d;
  }
  if (g != NULL) {
    if (alpha > 0) {
      *g = (-M_TWO_DIV_SQRT_PI*alpha*exp(-x*x) - pot)/d;
    } else {
      *g = -pot/d;
    }
    *g *= qprod/d; // compute derivative divided by d
  }
  pot *= qprod;
  return pot;
}
