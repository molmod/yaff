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
    (*result).rcut = 0.0;
    (*result).trunc_scheme = NULL;
  }
  return result;
}

void pair_pot_free(pair_pot_type *pair_pot) {
  free(pair_pot);
}

int pair_pot_ready(pair_pot_type *pair_pot) {
  return (*pair_pot).pair_data != NULL && (*pair_pot).pair_fn != NULL;
}

double pair_pot_get_rcut(pair_pot_type *pair_pot) {
  return (*pair_pot).rcut;
}

void pair_pot_set_rcut(pair_pot_type *pair_pot, double rcut) {
  (*pair_pot).rcut = rcut;
}

void pair_pot_set_trunc_scheme(pair_pot_type *pair_pot, trunc_scheme_type *trunc_scheme) {
  (*pair_pot).trunc_scheme = trunc_scheme;
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
    if (nlist[i].d < (*pair_pot).rcut) {
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
          // If a truncation scheme is defined, apply it.
          if ((*pair_pot).trunc_scheme!=NULL) {
            v *= (*(*pair_pot).trunc_scheme).trunc_fn(nlist[i].d, (*pair_pot).rcut, (*(*pair_pot).trunc_scheme).par, NULL);
          }
        } else {
          // Call the potential function with vg argument.
          // vg is the derivative of the pair potential divided by the distance.
          v = (*pair_pot).pair_fn((*pair_pot).pair_data, center_index, other_index, nlist[i].d, &vg);
          // If a truncation scheme is defined, apply it.
          if ((*pair_pot).trunc_scheme!=NULL) {
            // hg is (a pointer to) the derivative of the truncation function.
            h = (*(*pair_pot).trunc_scheme).trunc_fn(nlist[i].d, (*pair_pot).rcut, (*(*pair_pot).trunc_scheme).par, &hg);
            // chain rule:
            vg = vg*h + v*hg/nlist[i].d;
            v *= h;
          }
          vg *= s;
          if (gpos!=NULL) {
            h = nlist[i].dx*vg;
            gpos[3*other_index  ] += h;
            gpos[3*center_index   ] -= h;
            h = nlist[i].dy*vg;
            gpos[3*other_index+1] += h;
            gpos[3*center_index +1] -= h;
            h = nlist[i].dz*vg;
            gpos[3*other_index+2] += h;
            gpos[3*center_index +2] -= h;
          }
          if (vtens!=NULL) {
            vtens[0] += nlist[i].dx*nlist[i].dx*vg;
            vtens[4] += nlist[i].dy*nlist[i].dy*vg;
            vtens[8] += nlist[i].dz*nlist[i].dz*vg;
            h = nlist[i].dx*nlist[i].dy*vg;
            vtens[1] += h;
            vtens[3] += h;
            h = nlist[i].dx*nlist[i].dz*vg;
            vtens[2] += h;
            vtens[6] += h;
            h = nlist[i].dy*nlist[i].dz*vg;
            vtens[5] += h;
            vtens[7] += h;
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
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_lj;
    (*pair_data).sigma = sigma;
    (*pair_data).epsilon = epsilon;
  }
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




void pair_data_mm3_init(pair_pot_type *pair_pot, double *sigma, double *epsilon) {
  pair_data_mm3_type *pair_data;
  pair_data = malloc(sizeof(pair_data_mm3_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_mm3;
    (*pair_data).sigma = sigma;
    (*pair_data).epsilon = epsilon;
  }
}

double pair_fn_mm3(void *pair_data, long center_index, long other_index, double d, double *g) {
// E = epsilon*[1.84e5*exp(-12.0*R/sigma) - 2.25(sigma/R)^6]
  double sigma, epsilon, x, exponent;
  sigma = 0.5*(
    (*(pair_data_mm3_type*)pair_data).sigma[center_index]+
    (*(pair_data_mm3_type*)pair_data).sigma[other_index]
  );
  epsilon = sqrt(
    (*(pair_data_mm3_type*)pair_data).epsilon[center_index]*
    (*(pair_data_mm3_type*)pair_data).epsilon[other_index]
  );
  x = sigma/d;
  exponent = 1.84e5*exp(-12.0/x);
  x *= x;
  x *= 2.25*x*x;
  if (g != NULL) {
    *g =epsilon/d*(-12.0/sigma*exponent+6.0/d*x);
  }
  return epsilon*(exponent-x);
}



void pair_data_grimme_init(pair_pot_type *pair_pot, double *r0, double *c6) {
  pair_data_grimme_type *pair_data;
  pair_data = malloc(sizeof(pair_data_grimme_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_grimme;
    (*pair_data).r0 = r0;
    (*pair_data).c6 = c6;
  }
}

double pair_fn_grimme(void *pair_data, long center_index, long other_index, double d, double *g) {
// E = -1.1*damp(r)*c6/r**6 met damp(r)=1.0/(1.0+exp(-20*(r/r0-1.0))) [Grimme2006]
  double r0, c6, exponent, f, d6, e;
  r0 = (
    (*(pair_data_grimme_type*)pair_data).r0[center_index]+
    (*(pair_data_grimme_type*)pair_data).r0[other_index]
  );
  c6 = sqrt(
    (*(pair_data_grimme_type*)pair_data).c6[center_index]*
    (*(pair_data_grimme_type*)pair_data).c6[other_index]
  );
  exponent = exp(-20.0*(d/r0-1.0));
  f = 1.0/(1.0+exponent);
  d6 = d*d*d;
  d6 *= d6;
  e = 1.1*f*c6/d6;
  if (g != NULL) {
    *g = e/d*(6.0/d-20.0/r0*f*exponent);
  }
  return -e;
}



void pair_data_exprep_init(pair_pot_type *pair_pot, double *amp, int amp_mix, double amp_mix_coeff, double *b, int b_mix, double b_mix_coeff) {
  pair_data_exprep_type *pair_data;
  pair_data = malloc(sizeof(pair_data_exprep_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_exprep;
    (*pair_data).amp = amp;
    (*pair_data).amp_mix = amp_mix;
    (*pair_data).amp_mix_coeff = amp_mix_coeff;
    (*pair_data).b = b;
    (*pair_data).b_mix = b_mix;
    (*pair_data).b_mix_coeff = b_mix_coeff;
  }
}

double pair_fn_exprep(void *pair_data, long center_index, long other_index, double d, double *g) {
  double amp0, amp1, amp, b, e;
  pair_data_exprep_type *pd;
  pd = (pair_data_exprep_type*)pair_data;
  amp0 = (*pd).amp[center_index];
  amp1 = (*pd).amp[other_index];
  switch ((*pd).amp_mix) {
    case 0:
      amp = sqrt(amp0*amp1);
      break;
    case 1:
      amp = (log(amp0)+log(amp1))/2;
      amp *= 1 - ((*pd).amp_mix_coeff)*fabs(log(amp0/amp1));
      amp = exp(amp);
      break;
    default:
      amp = FP_NAN;
  }
  switch ((*pd).b_mix) {
    case 0:
      b = ((*pd).b[center_index]+(*pd).b[other_index])/2;
      break;
    case 1:
      b = ((*pd).b[center_index]+(*pd).b[other_index])/2;
      b *= 1 - ((*pd).b_mix_coeff)*fabs(log(amp0/amp1));
      break;
    default:
      b = FP_NAN;
  }
  e = amp*exp(-b*d);
  if (g != NULL) {
    *g = -e*b/d;
  }
  return e;
}

int pair_data_exprep_get_amp_mix(pair_pot_type *pair_pot) {
  return (*(pair_data_exprep_type*)((*pair_pot).pair_data)).amp_mix;
}

double pair_data_exprep_get_amp_mix_coeff(pair_pot_type *pair_pot) {
  return (*(pair_data_exprep_type*)((*pair_pot).pair_data)).amp_mix_coeff;
}

int pair_data_exprep_get_b_mix(pair_pot_type *pair_pot) {
  return (*(pair_data_exprep_type*)((*pair_pot).pair_data)).b_mix;
}

double pair_data_exprep_get_b_mix_coeff(pair_pot_type *pair_pot){
  return (*(pair_data_exprep_type*)((*pair_pot).pair_data)).b_mix_coeff;
}



void pair_data_dampdisp_init(pair_pot_type *pair_pot, double *c6, double *b, double *vol) {
  pair_data_dampdisp_type *pair_data;
  pair_data = malloc(sizeof(pair_data_dampdisp_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_dampdisp;
    (*pair_data).c6 = c6;
    (*pair_data).b = b;
    (*pair_data).vol = vol;
  }
}

double tang_toennies(double x, int order, double *g){
  double tmp, poly, last, e;
  int k, fac;
  poly = 0.0;
  fac = 1;
  tmp = 1.0;
  for (k=0; k<order; k++) {
    poly += tmp/fac;
    fac *= k+1;
    tmp *= x;
  }
  last = tmp/fac;
  poly += last;
  e = exp(-x);
  if (g != NULL) {
    *g = e*last;
  }
  return 1.0 - poly*e;
}

double pair_fn_dampdisp(void *pair_data, long center_index, long other_index, double d, double *g) {
  double c60, c61, ratio, c6, b, disp, damp;
  // Load parameters from data structure and mix
  pair_data_dampdisp_type *pd;
  pd = (pair_data_dampdisp_type*)pair_data;
  c60 = (*pd).c6[center_index];
  c61 = (*pd).c6[other_index];
  ratio = (*pd).vol[center_index]/(*pd).vol[other_index];
  ratio *= ratio;
  c6 = 2.0*c60*c61/(c60*ratio+c61/ratio);
  b = 0.5*((*pd).b[center_index]+(*pd).b[other_index]);
  // compute the damping
  damp = tang_toennies(b*d, 6, g);
  // compute the energy
  disp = d*d;
  disp *= disp*disp;
  disp = -c6/disp;
  if (g != NULL) {
    *g = ((*g)*b-6.0/d*damp)*disp/d;
  }
  return damp*disp;
}



void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha) {
  pair_data_ei_type *pair_data;
  pair_data = malloc(sizeof(pair_data_ei_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_ei;
    (*pair_data).charges = charges;
    (*pair_data).alpha = alpha;
  }
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

double pair_data_ei_get_alpha(pair_pot_type *pair_pot) {
  return (*(pair_data_ei_type*)((*pair_pot).pair_data)).alpha;
}
