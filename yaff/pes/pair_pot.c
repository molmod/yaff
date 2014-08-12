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


#include <math.h>
#include <stdlib.h>
#include "constants.h"
#include "pair_pot.h"
#include <stdio.h>

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

double get_scaling(scaling_row_type *stab, long a, long b, long *row, long size) {
  if (*row >= size) return 1.0;
  while (stab[*row].a < a) {
    (*row)++;
    if (*row >= size) return 1.0;
  }
  if (stab[*row].a != a) return 1.0;
  while (stab[*row].b < b) {
    (*row)++;
    if (*row >= size) return 1.0;
    if (stab[*row].a != a) return 1.0;
  }
  if ((stab[*row].b == b) && (stab[*row].a == a)) {
    return stab[*row].scale;
  }
  return 1.0;
}


double pair_pot_compute(neigh_row_type *neighs,
                        long nneigh, scaling_row_type *stab,
                        long nstab, pair_pot_type *pair_pot,
                        double *gpos, double* vtens) {
  long i, srow, center_index, other_index;
  double s, energy, v, vg, h, hg;
  double delta[3], vg_cart[3];
  energy = 0.0;
  // Reset the row counter for the scaling.
  srow = 0;
  // Compute the interactions.
  for (i=0; i<nneigh; i++) {
    // Find the scale
    if (neighs[i].d < (*pair_pot).rcut) {
      center_index = neighs[i].a;
      other_index = neighs[i].b;
      if ((neighs[i].r0 == 0) && (neighs[i].r1 == 0) && (neighs[i].r2 == 0)) {
        s = get_scaling(stab, center_index, other_index, &srow, nstab);
      } else {
        s = 1.0;
      }
      // If the scale is non-zero, compute the contribution.
      if (s > 0.0) {
        //Construct vector of distances, needed for some pair potentials
        delta[0] = neighs[i].dx;
        delta[1] = neighs[i].dy;
        delta[2] = neighs[i].dz;
        if ((gpos==NULL) && (vtens==NULL)) {
          // Call the potential function without g argument.
          v = (*pair_pot).pair_fn((*pair_pot).pair_data, center_index, other_index, neighs[i].d, delta, NULL, NULL);
          // If a truncation scheme is defined, apply it.
          if (((*pair_pot).trunc_scheme!=NULL) && (v!=0.0)) {
            v *= (*(*pair_pot).trunc_scheme).trunc_fn(neighs[i].d, (*pair_pot).rcut, (*(*pair_pot).trunc_scheme).par, NULL);
          }
        } else {
          // Call the potential function with vg argument.
          // vg is the derivative of the pair potential divided by the distance.
          // vg_cart contains the (partial) derivatives of the pair potential to
          // cartesian coordinates. Implicit dependence of the pair potential on
          // cartesian coordinates is captured by vg.
          vg_cart[0] = 0.0; //vg_cart is reset here because not all pair_fn set it.
          vg_cart[1] = 0.0;
          vg_cart[2] = 0.0;
          v = (*pair_pot).pair_fn((*pair_pot).pair_data, center_index, other_index, neighs[i].d, delta, &vg, vg_cart);
          // If a truncation scheme is defined, apply it.
          if (((*pair_pot).trunc_scheme!=NULL) && ((v!=0.0) || (vg!=0.0))) {
            // hg is (a pointer to) the derivative of the truncation function.
            h = (*(*pair_pot).trunc_scheme).trunc_fn(neighs[i].d,    (*pair_pot).rcut, (*(*pair_pot).trunc_scheme).par, &hg);
            // chain rule:
            vg = vg*h + v*hg/neighs[i].d;
            vg_cart[0] = vg_cart[0]*h;
            vg_cart[1] = vg_cart[1]*h;
            vg_cart[2] = vg_cart[2]*h;
            v *= h;
          }
          //printf("C %3i %3i (% 3i % 3i % 3i) %10.7f %3.1f %10.3e\n", center_index, other_index, neighs[i].r0, neighs[i].r1, neighs[i].r2, neighs[i].d, s, s*v);
          vg *= s;
          vg_cart[0] *= s;
          vg_cart[1] *= s;
          vg_cart[2] *= s;
          if (gpos!=NULL) {
            h = neighs[i].dx*vg;
            gpos[3*other_index  ] += h + vg_cart[0];
            gpos[3*center_index   ] -= h + vg_cart[0];
            h = neighs[i].dy*vg;
            gpos[3*other_index+1] += h + vg_cart[1];
            gpos[3*center_index +1] -= h + vg_cart[1];
            h = neighs[i].dz*vg;
            gpos[3*other_index+2] += h + vg_cart[2];
            gpos[3*center_index +2] -= h + vg_cart[2];
          }
          if (vtens!=NULL) {
            vtens[0] += neighs[i].dx*(neighs[i].dx*vg+vg_cart[0]);
            vtens[4] += neighs[i].dy*(neighs[i].dy*vg+vg_cart[1]);
            vtens[8] += neighs[i].dz*(neighs[i].dz*vg+vg_cart[2]);
            //h = neighs[i].dx*(neighs[i].dy*vg+vg_cart[1]);
            vtens[1] += neighs[i].dx*(neighs[i].dy*vg+vg_cart[1]);
            vtens[3] += neighs[i].dy*(neighs[i].dx*vg+vg_cart[0]);
            //h = neighs[i].dx*(neighs[i].dz*vg+vg_cart[2]);
            vtens[2] += neighs[i].dx*(neighs[i].dz*vg+vg_cart[2]);
            vtens[6] += neighs[i].dz*(neighs[i].dx*vg+vg_cart[0]);
            //h = neighs[i].dy*(neighs[i].dz*vg+vg_cart[2]);
            vtens[5] += neighs[i].dy*(neighs[i].dz*vg+vg_cart[2]);
            vtens[7] += neighs[i].dz*(neighs[i].dy*vg+vg_cart[1]);
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

double pair_fn_lj(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  double sigma, epsilon, x;
  sigma = 0.5*(
    (*(pair_data_lj_type*)pair_data).sigma[center_index]+
    (*(pair_data_lj_type*)pair_data).sigma[other_index]
  );
    // Uncomment for geometric mixing rules
    /*
  sigma = sqrt(
    (*(pair_data_lj_type*)pair_data).sigma[center_index]*
    (*(pair_data_lj_type*)pair_data).sigma[other_index]
  );
    */
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




void pair_data_mm3_init(pair_pot_type *pair_pot, double *sigma, double *epsilon, int *onlypauli) {
  pair_data_mm3_type *pair_data;
  pair_data = malloc(sizeof(pair_data_mm3_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_mm3;
    (*pair_data).sigma = sigma;
    (*pair_data).epsilon = epsilon;
    (*pair_data).onlypauli = onlypauli;
  }
}

double pair_fn_mm3(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
// E = epsilon*[1.84e5*exp(-12.0*R/sigma) - 2.25(sigma/R)^6]
  double sigma, epsilon, x, exponent;
  int onlypauli;
  sigma = (
    (*(pair_data_mm3_type*)pair_data).sigma[center_index]+
    (*(pair_data_mm3_type*)pair_data).sigma[other_index]
  );
  epsilon = sqrt(
    (*(pair_data_mm3_type*)pair_data).epsilon[center_index]*
    (*(pair_data_mm3_type*)pair_data).epsilon[other_index]
  );
  onlypauli = (
    (*(pair_data_mm3_type*)pair_data).onlypauli[center_index]+
    (*(pair_data_mm3_type*)pair_data).onlypauli[other_index]
  );
  x = sigma/d;
  exponent = 1.84e5*exp(-12.0/x);
  if (onlypauli == 0) {
    x *= x;
    x *= 2.25*x*x;
    if (g != NULL) {
      *g =epsilon/d*(-12.0/sigma*exponent+6.0/d*x);
    }
    return epsilon*(exponent-x);
  } else {
    if (g != NULL) {
        *g =epsilon/d*(-12.0/sigma*exponent);
    }
    return epsilon*exponent;
  }
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

double pair_fn_grimme(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
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



void pair_data_exprep_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *amp_cross, double *b_cross) {
  pair_data_exprep_type *pair_data;
  pair_data = malloc(sizeof(pair_data_exprep_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_exprep;
    (*pair_data).nffatype = nffatype;
    (*pair_data).ffatype_ids = ffatype_ids;
    (*pair_data).amp_cross = amp_cross;
    (*pair_data).b_cross = b_cross;
  }
}

double pair_fn_exprep(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  long i;
  double amp, b, e;
  pair_data_exprep_type *pd;
  pd = (pair_data_exprep_type*)pair_data;
  i = (*pd).ffatype_ids[center_index]*(*pd).nffatype + (*pd).ffatype_ids[other_index];
  amp = (*pd).amp_cross[i];
  if (amp==0.0) goto bail;
  b = (*pd).b_cross[i];
  if (b==0.0) goto bail;
  e = amp*exp(-b*d);
  if (g != NULL) *g = -e*b/d;
  return e;
bail:
  if (g != NULL) *g = 0.0;
  return 0.0;
}



void pair_data_dampdisp_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *c6_cross, double *b_cross) {
  pair_data_dampdisp_type *pair_data;
  pair_data = malloc(sizeof(pair_data_dampdisp_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_dampdisp;
    (*pair_data).nffatype = nffatype;
    (*pair_data).ffatype_ids = ffatype_ids;
    (*pair_data).c6_cross = c6_cross;
    (*pair_data).b_cross = b_cross;
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

double pair_fn_dampdisp(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  long i;
  double b, disp, damp, c6;
  // Load parameters from data structure and mix
  pair_data_dampdisp_type *pd;
  pd = (pair_data_dampdisp_type*)pair_data;
  i = (*pd).ffatype_ids[center_index]*(*pd).nffatype + (*pd).ffatype_ids[other_index];
  c6 = (*pd).c6_cross[i];
  if (c6==0.0) return 0.0;
  b = (*pd).b_cross[i];
  if (b==0.0) {
    // without damping
    disp = d*d;
    disp *= disp*disp;
    disp = -c6/disp;
    if (g != NULL) {
      *g = -6.0*disp/(d*d);
    }
    return disp;
  } else {
    // with damping
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
}


void pair_data_disp68bjdamp_init(pair_pot_type *pair_pot, long nffatype, long* ffatype_ids, double *c6_cross, double *c8_cross, double *R_cross, double c6_scale, double c8_scale, double bj_a, double bj_b) {
  pair_data_disp68bjdamp_type *pair_data;
  pair_data = malloc(sizeof(pair_data_disp68bjdamp_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_disp68bjdamp;
    (*pair_data).nffatype = nffatype;
    (*pair_data).ffatype_ids = ffatype_ids;
    (*pair_data).c6_cross = c6_cross;
    (*pair_data).c8_cross = c8_cross;
    (*pair_data).R_cross = R_cross;
    (*pair_data).c6_scale = c6_scale;
    (*pair_data).c8_scale = c8_scale;
    (*pair_data).bj_a = bj_a;
    (*pair_data).bj_b = bj_b;
  }
}

double pair_fn_disp68bjdamp(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  long i;
  double c6, c8, R, R2, R4, R6, R8, d2, d4, d6, d8;
  // Load parameters from data structure
  pair_data_disp68bjdamp_type *pd;
  pd = (pair_data_disp68bjdamp_type*)pair_data;
  i = (*pd).ffatype_ids[center_index]*(*pd).nffatype + (*pd).ffatype_ids[other_index];
  c6 = (*pd).c6_cross[i];
  c8 = (*pd).c8_cross[i];
  R  = (*pd).bj_a * (*pd).R_cross[i] + (*pd).bj_b;
  // Compute succesive powers of distance and R
  R2 = R*R;
  R4 = R2*R2;
  R6 = R4*R2;
  R8 = R4*R4;
  d2 = d*d;
  d4 = d2*d2;
  d6 = d4*d2;
  d8 = d4*d4;
  // Add everything together
  double pot = -(*pd).c6_scale*c6/(d6+R6)-(*pd).c8_scale*c8/(d8+R8);
  //printf("%d %d %f %f %f %f %f %f %f\n",center_index,other_index,c6,c8,d,R,pot,(*pd).c6_scale,(*pd).c8_scale);
  if (g != NULL) {
    *g = 6.0*(*pd).c6_scale*c6/(d6+R6)/(d6+R6)*d4 + 8.0*(*pd).c8_scale*c8/(d8+R8)/(d8+R8)*d6;
  }
  return pot;
}

double pair_data_disp68bjdamp_get_c6_scale(pair_pot_type *pair_pot) {
  return (*(pair_data_disp68bjdamp_type*)((*pair_pot).pair_data)).c6_scale;
}

double pair_data_disp68bjdamp_get_c8_scale(pair_pot_type *pair_pot){
  return (*(pair_data_disp68bjdamp_type*)((*pair_pot).pair_data)).c8_scale;
}

double pair_data_disp68bjdamp_get_bj_a(pair_pot_type *pair_pot){
  return (*(pair_data_disp68bjdamp_type*)((*pair_pot).pair_data)).bj_a;
}

double pair_data_disp68bjdamp_get_bj_b(pair_pot_type *pair_pot){
  return (*(pair_data_disp68bjdamp_type*)((*pair_pot).pair_data)).bj_b;
}


void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha, double dielectric, double *radii) {
  pair_data_ei_type *pair_data;
  pair_data = malloc(sizeof(pair_data_ei_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_ei;
    (*pair_data).charges = charges;
    (*pair_data).alpha = alpha;
    (*pair_data).dielectric = dielectric;
    (*pair_data).radii = radii;
  }
}

double pair_fn_ei(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  double pot, alpha, qprod, x, y, r_ab;
  qprod = (
    (*(pair_data_ei_type*)pair_data).charges[center_index]*
    (*(pair_data_ei_type*)pair_data).charges[other_index]
  ) / (*(pair_data_ei_type*)pair_data).dielectric;
  //Averaged atomic radius needed to compute EI interaction between Gaussian charge distributions
  //TODO: store square of atomic radii
  //TODO: deal with missing radii
  r_ab = sqrt( (*(pair_data_ei_type*)pair_data).radii[center_index] * (*(pair_data_ei_type*)pair_data).radii[center_index] +
               (*(pair_data_ei_type*)pair_data).radii[other_index] * (*(pair_data_ei_type*)pair_data).radii[other_index] );
  alpha = (*(pair_data_ei_type*)pair_data).alpha;

  //Original minus gaussian screening charge distribution
  if (alpha > 0) {
    if (r_ab > 0 ){ //Original is a Gaussian charge distribution
      x = alpha*d;
      y = d/r_ab;
      pot = qprod/d*(erfc(x) - erfc(y) );
      if (g != NULL) *g = ( M_TWO_DIV_SQRT_PI*(exp(-y*y)/r_ab - exp(-x*x)*alpha)*qprod - pot)/(d*d);
    }
    else{ //Original is a point monopole
      x = alpha*d;
      pot = qprod*erfc(x)/d;
      if (g != NULL) *g = (-M_TWO_DIV_SQRT_PI*alpha*exp(-x*x)*qprod - pot)/(d*d);
    }
  }
  //Original only
  else {
    if (r_ab > 0) {//Original is a gaussian charge distribution
         y = d/r_ab;
         pot = qprod/d*erf(y);
         if (g != NULL) *g = (M_TWO_DIV_SQRT_PI/r_ab*exp(-y*y)*qprod-pot)/(d*d);
     }
     else{ //Original is a point monopole
         pot = qprod/d;
         if (g != NULL) *g = -pot/(d*d);
     }
  }
  return pot;
}

double pair_data_ei_get_alpha(pair_pot_type *pair_pot) {
  return (*(pair_data_ei_type*)((*pair_pot).pair_data)).alpha;
}

double pair_data_ei_get_dielectric(pair_pot_type *pair_pot) {
  return (*(pair_data_ei_type*)((*pair_pot).pair_data)).dielectric;
}


void pair_data_eidip_init(pair_pot_type *pair_pot, double *charges, double *dipoles, double alpha, double *radii, double *radii2) {
  pair_data_eidip_type *pair_data;
  pair_data = malloc(sizeof(pair_data_eidip_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_eidip;
    (*pair_data).charges = charges;
    (*pair_data).dipoles = dipoles;
    (*pair_data).alpha = alpha;
    (*pair_data).radii = radii;
    (*pair_data).radii2 = radii2;
  }
}

double pair_fn_eidip(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  double pot;
  double qi, qj, dix, diy, diz, djx, djy, djz;
  double x, d_2, alpha, fac0, fac1, fac2, fac3;
  double mui_dot_delta, muj_dot_delta, mui_dot_muj;
  double r_qq, r_qd, r_dq, r_dd;
  double fac0_qq, fac1_qq, fac1_qd, fac2_qd, fac1_dq, fac2_dq, fac1_dd, fac2_dd, fac3_dd;

  //Averaged atomic radius needed to compute EI interaction between Gaussian charge distributions
  //TODO: store square of atomic radii
  //TODO: deal with missing radii
  r_qq = sqrt( (*(pair_data_eidip_type*)pair_data).radii[center_index] * (*(pair_data_eidip_type*)pair_data).radii[center_index] +
               (*(pair_data_eidip_type*)pair_data).radii[other_index ] * (*(pair_data_eidip_type*)pair_data).radii[other_index] );
  r_qd = sqrt( (*(pair_data_eidip_type*)pair_data).radii[center_index] * (*(pair_data_eidip_type*)pair_data).radii[center_index] +
               (*(pair_data_eidip_type*)pair_data).radii2[other_index ] * (*(pair_data_eidip_type*)pair_data).radii2[other_index] );
  r_dq = sqrt( (*(pair_data_eidip_type*)pair_data).radii2[center_index] * (*(pair_data_eidip_type*)pair_data).radii2[center_index] +
               (*(pair_data_eidip_type*)pair_data).radii[other_index ] * (*(pair_data_eidip_type*)pair_data).radii[other_index] );
  r_dd = sqrt( (*(pair_data_eidip_type*)pair_data).radii2[center_index] * (*(pair_data_eidip_type*)pair_data).radii2[center_index] +
               (*(pair_data_eidip_type*)pair_data).radii2[other_index ] * (*(pair_data_eidip_type*)pair_data).radii2[other_index] );

  //Charges
  qi = (*(pair_data_eidip_type*)pair_data).charges[center_index];
  qj = (*(pair_data_eidip_type*)pair_data).charges[other_index];
  //Dipole components
  dix = (*(pair_data_eidip_type*)pair_data).dipoles[ 3*center_index  + 0 ];
  diy = (*(pair_data_eidip_type*)pair_data).dipoles[ 3*center_index  + 1 ];
  diz = (*(pair_data_eidip_type*)pair_data).dipoles[ 3*center_index  + 2 ];
  djx = (*(pair_data_eidip_type*)pair_data).dipoles[ 3*other_index   + 0 ];
  djy = (*(pair_data_eidip_type*)pair_data).dipoles[ 3*other_index   + 1 ];
  djz = (*(pair_data_eidip_type*)pair_data).dipoles[ 3*other_index   + 2 ];
  //Some useful definitions
  d_2 = 1.0/(d*d);
  alpha = (*(pair_data_eidip_type*)pair_data).alpha;
  mui_dot_delta = dix*delta[0] + diy*delta[1] + diz*delta[2];
  muj_dot_delta = djx*delta[0] + djy*delta[1] + djz*delta[2];
  mui_dot_muj = dix*djx + diy*djy + diz*djz;
  //Original minus gaussian screening charge distribution
  if (alpha > 0){
    x = alpha*d;
    fac0 = erfc(x)/d;
    fac1 = ( fac0 + M_TWO_DIV_SQRT_PI*alpha*exp(-x*x))*d_2;
    fac2 = (3.0*fac1 + 2.0*M_TWO_DIV_SQRT_PI*alpha*alpha*alpha*exp(-x*x))*d_2;
    if (g != NULL){
      fac3 = (5.0*fac2 + 4.0*M_TWO_DIV_SQRT_PI*alpha*alpha*alpha*alpha*alpha*exp(-x*x))*d_2;}
  }
  //Original only
  else{
    fac0 = 1.0/d;
    fac1 = fac0*d_2;
    fac2 = 3.0*fac1*d_2;
    if (g != NULL ){
       fac3 = 5.0*fac2*d_2;}
  }

    if (r_qq > 0) { //Correction for gaussian charge distribution
       x = d/r_qq;
       fac0_qq = erfc(x)/d;
       fac1_qq = ( fac0_qq + M_TWO_DIV_SQRT_PI/r_qq*exp(-x*x))*d_2;
    }else{fac0_qq = 0.0; fac1_qq = 0;}

    if (r_qd > 0) { //Correction for gaussian charge distribution
       x = d/r_qd;
       fac1_qd = ( erfc(x)/d + M_TWO_DIV_SQRT_PI/r_qd*exp(-x*x))*d_2;
       fac2_qd = (3.0*fac1_qd + 2.0*M_TWO_DIV_SQRT_PI/r_qd/r_qd/r_qd*exp(-x*x))*d_2;
    }else{fac1_qd = 0.0; fac2_qd = 0;}

    if (r_dq > 0) { //Correction for gaussian charge distribution
       x = d/r_dq;
       fac1_dq = ( erfc(x)/d + M_TWO_DIV_SQRT_PI/r_dq*exp(-x*x))*d_2;
       fac2_dq = (3.0*fac1_dq + 2.0*M_TWO_DIV_SQRT_PI/r_dq/r_dq/r_dq*exp(-x*x))*d_2;
    }else{fac1_dq = 0.0; fac2_dq = 0;}

    if (r_dd > 0) { //Correction for gaussian charge distribution
       x = d/r_dd;
       fac1_dd = ( erfc(x)/d + M_TWO_DIV_SQRT_PI/r_dd*exp(-x*x))*d_2;
       fac2_dd = (3.0*fac1_dd + 2.0*M_TWO_DIV_SQRT_PI/r_dd/r_dd/r_dd*exp(-x*x))*d_2;
       fac3_dd = (5.0*fac2_dd + 4.0*M_TWO_DIV_SQRT_PI/r_dd/r_dd/r_dd/r_dd/r_dd*exp(-x*x))*d_2;
    }else{fac1_dd = 0.0; fac2_dd = 0; fac3_dd = 0.0;}


  pot = qi*qj*(fac0-fac0_qq); //CC interaction
  pot += fac1*(qi*muj_dot_delta-qj*mui_dot_delta); //CD and DC interaction
  pot += -fac1_qd*qi*muj_dot_delta + fac1_dq*qj*mui_dot_delta; //CD and DC interaction
  pot += (fac1-fac1_dd)*mui_dot_muj  - (fac2-fac2_dd)*muj_dot_delta*mui_dot_delta; //DD interaction
  if (g != NULL ){
      *g  = -qi*qj*(fac1-fac1_qq); //CC interaction
      *g += -fac2*(qi*muj_dot_delta - qj*mui_dot_delta); //CD and DC interaction
      *g += fac2_qd*qi*muj_dot_delta - fac2_dq*qj*mui_dot_delta; //CD and DC interaction
      *g += -(fac2-fac2_dd)*mui_dot_muj + (fac3-fac3_dd)*muj_dot_delta*mui_dot_delta; //DD interaction
      //CC and DC interaction
      g_cart[0] = fac1*(qi*djx-qj*dix);
      g_cart[0] += -fac1_qd*qi*djx + fac1_dq*qj*dix;
      g_cart[1] = fac1*(qi*djy-qj*diy);
      g_cart[1] += -fac1_qd*qi*djy + fac1_dq*qj*diy;
      g_cart[2] = fac1*(qi*djz-qj*diz);
      g_cart[2] += -fac1_qd*qi*djz + fac1_dq*qj*diz;
      //DD interaction
      g_cart[0] += - (fac2-fac2_dd)*(dix*muj_dot_delta + djx*mui_dot_delta );
      g_cart[1] += - (fac2-fac2_dd)*(diy*muj_dot_delta + djy*mui_dot_delta );
      g_cart[2] += - (fac2-fac2_dd)*(diz*muj_dot_delta + djz*mui_dot_delta );
  }
  return pot;
}

double pair_data_eidip_get_alpha(pair_pot_type *pair_pot) {
  return (*(pair_data_eidip_type*)((*pair_pot).pair_data)).alpha;
}

void pair_data_eislater1s1scorr_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double *slater1s_Z) {
  pair_data_eislater1s1scorr_type *pair_data;
  pair_data = malloc(sizeof(pair_data_eislater1s1scorr_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
      (*pair_pot).pair_fn = pair_fn_eislater1s1scorr;
      (*pair_data).widths = slater1s_widths;
      (*pair_data).N = slater1s_N;
      (*pair_data).Z = slater1s_Z;
  }
}

double pair_fn_eislater1s1scorr(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  double a, b, dab, Za, Na, Zb, Nb, da, db;
  double pot = 0.0;
  a  = (*(pair_data_eislater1s1scorr_type*)pair_data).widths[center_index];
  b  = (*(pair_data_eislater1s1scorr_type*)pair_data).widths[other_index];
  Na = (*(pair_data_eislater1s1scorr_type*)pair_data).N[center_index];
  Nb = (*(pair_data_eislater1s1scorr_type*)pair_data).N[other_index];
  Za = (*(pair_data_eislater1s1scorr_type*)pair_data).Z[center_index];
  Zb = (*(pair_data_eislater1s1scorr_type*)pair_data).Z[other_index];
  dab = a - b;
  da = d/a;
  db = d/b;
  // Discriminate between small and not small difference in slater width
  if (fabs(dab) > 0.025) {  // TODO: carefully check criterium
    double a2 = a*a;
    double a3 = a2*a;
    double a4 = a2*a2;
    double b2 = b*b;
    double b3 = b2*b;
    double b4 = b2*b2;
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    double pot1 = Na*(Zb*(1.0+0.5*da) + Nb*( a4*(a2-3.0*b2)*diff3 + 0.5*a3*diff2*d )) * exp(-da) / d;
    double pot2 = Nb*(Za*(1.0+0.5*db) + Na*( b4*(3.0*a2-b2)*diff3 + 0.5*b3*diff2*d )) * exp(-db) / d;
    pot += - pot1 - pot2;
    if (g != NULL) *g = (-pot/d+pot1/a+pot2/b - Na*(0.5*Zb/a + Nb*0.5*a3*diff2)*exp(-da)/d - Nb*(0.5*Za/b + Na*0.5*b3*diff2)*exp(-db)/d)/d;
   } else {
    double da2 = da*da;
    double da3 = da2*da;
    double da4 = da2*da2;
    double a2i = 1.0/(a*a);
    double a3i = a2i/a;
    double a4i = a2i*a2i;
    pot -= Na*Zb*(1.0+0.5*da)* exp(-da) / d;
    pot -= Nb*Za*(1.0+0.5*db)* exp(-db) / d;
    pot -= Na*Nb*(48.0+33.0*da+9.0*da2+da3)*exp(-da)/48.0/d; // Taylor 0th order in dab
    pot += Na*Nb*(15.0+15.0*da+6.0*da2+da3)*exp(-da)*dab/96.0*a2i; // Taylor 1st order in dab
    pot -= Na*Nb*(-60.0-60.0*da-15.0*da2+5.0*da3+3.0*da4)*exp(-da)*dab*dab/960.0*a3i; // Taylor 2nd order in dab
    if (g != NULL) {
      *g  = Na*Zb*(1.0+0.5*da)* exp(-da) / d * (1.0/d + 1.0/a) / d;
      *g += Nb*Za*(1.0+0.5*db)* exp(-db) / d * (1.0/d + 1.0/b) / d;
      *g += Na*Nb*(48.0+33.0*da+9.0*da2+da3)*exp(-da)/48.0/d * (1.0/d + 1.0/a) / d;
      *g -= Na*Nb*(33.0+18.0*da+3.0*da2)*exp(-da)/48.0/d/d/a;
      *g -= Na*Nb*(15.0+15.0*da+6.0*da2+da3)*exp(-da)*dab/96.0*a3i/d;
      *g += Na*Nb*(15.0+12.0*da+3.0*da2)*exp(-da)*dab/96.0*a3i/d;
      *g += Na*Nb*(-60.0-60.0*da-15.0*da2+5.0*da3+3.0*da4)*exp(-da)*dab*dab/960.0*a4i/d;
      *g -= Na*Nb*(-60.0-30.0*da+15.0*da2+12.0*da3)*exp(-da)*dab*dab/960.0*a4i/d;
    }
  }
  return pot;
}


double slater_00_00_olp(double a, double b, double d, double *g){
  // Overlap between two 1s (=00) Slater densities carrying unit charge
  // a: width of first Slater
  // b: width of second Slater
  // d: distance between two centers
  // g: pointer to gradient
  // Returns
  double dab, da, db;
  double pot = 0.0;
  dab = a - b;
  da = d/a;
  db = d/b;
  // Discriminate between small and not small difference in slater width
  if (fabs(dab) > 0.025) {
    double a2 = a*a;
    double b2 = b*b;
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    double pot1 = 0.5*(-4.0*a2*b2*diff3 + a*d*diff2)*exp(-da)/d/M_FOUR_PI;
    double pot2 = 0.5*( 4.0*a2*b2*diff3 + b*d*diff2)*exp(-db)/d/M_FOUR_PI;
    pot += pot1 + pot2;
    if (g != NULL) {
      *g = -pot/d/d-pot1/d/a-pot2/d/b + 0.5*a*diff2*exp(-da)/d/M_FOUR_PI/d + 0.5*b*diff2*exp(-db)/d/M_FOUR_PI/d;
    }
  } else {
    double da2 = da*da;
    double da3 = da2*da;
    double da4 = da2*da2;
    double a2i = 1.0/(a*a);
    double a3i = a2i/a;
    double a4i = a2i*a2i;
    double a5i = a3i*a2i;
    double a6i = a3i*a3i;
    pot += (da2+3.0*da+3.0)*exp(-da)*a3i/48.0/M_FOUR_PI;
    pot += (-da3+2.0*da2+9.0*da+9.0)*exp(-da)*a4i/96.0/M_FOUR_PI*dab;
    pot += (3.0*da4-25.0*da3+5.0*da2+90.0*da+90.0)*exp(-da)*a5i/960.0/M_FOUR_PI*dab*dab;
    if (g != NULL) {
        *g  = -pot/d/a;
        *g += (3.0+2.0*da)*exp(-da)*a4i/48.0/M_FOUR_PI/d;
        *g += (9.0+4.0*da-3.0*da2)*exp(-da)*a5i/96.0/M_FOUR_PI*dab/d;
        *g += (90.0+10.0*da-75.0*da2+12.0*da3)*exp(-da)*a6i/960.0/M_FOUR_PI*dab*dab;
    }
  }
  return pot;
}


void pair_data_olpslater1s1s_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double ex_scale, double corr_a, double corr_b, double corr_c ) {
  pair_data_olpslater1s1s_type *pair_data;
  pair_data = malloc(sizeof(pair_data_olpslater1s1s_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_olpslater1s1s;
    (*pair_data).widths = slater1s_widths;
    (*pair_data).N = slater1s_N;
    (*pair_data).ex_scale = ex_scale;
    (*pair_data).corr_a = corr_a;
    (*pair_data).corr_b = corr_b;
    (*pair_data).corr_c = corr_c;
  }
}


double pair_fn_olpslater1s1s(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  double a, b, Na, Nb;
  double pot = 0.0;
  a  = (*(pair_data_olpslater1s1s_type*)pair_data).widths[center_index];
  b  = (*(pair_data_olpslater1s1s_type*)pair_data).widths[other_index];
  Na = (*(pair_data_olpslater1s1s_type*)pair_data).N[center_index];
  Nb = (*(pair_data_olpslater1s1s_type*)pair_data).N[other_index];
  // Overlap between unit Slater densities
  pot += slater_00_00_olp(a, b, d, g);
  // Multiply with scaling factor and populations
  pot *= Na*Nb*(*(pair_data_olpslater1s1s_type*)pair_data).ex_scale;
  if (g != NULL) *g *= Na*Nb*(*(pair_data_olpslater1s1s_type*)pair_data).ex_scale;
  // Apply corrections to the overlap expression
  double ca = (*(pair_data_olpslater1s1s_type*)pair_data).corr_a;
  double cb = (*(pair_data_olpslater1s1s_type*)pair_data).corr_b;
  double cc = (*(pair_data_olpslater1s1s_type*)pair_data).corr_c;
  if ( cc != 0.0 ) pot *= 1.0 + cc*(Na+Nb);
  if ( ca != 0.0 ) pot *= 1.0 - exp(ca-cb*d/sqrt(a*b));
  if (g != NULL) {
    if ( cc != 0.0 ) *g *= 1.0 + cc*(Na+Nb);
    if ( ca != 0.0 ) {
      *g *= 1.0 - exp(ca-cb*d/sqrt(a*b));
      *g += pot*cb/sqrt(a*b)*exp(ca-cb*d/sqrt(a*b))/d;
    }
  }
  return pot;
}

double pair_data_olpslater1s1s_get_ex_scale(pair_pot_type *pair_pot) {
  return (*(pair_data_olpslater1s1s_type*)((*pair_pot).pair_data)).ex_scale;
}

double pair_data_olpslater1s1s_get_corr_a(pair_pot_type *pair_pot) {
  return (*(pair_data_olpslater1s1s_type*)((*pair_pot).pair_data)).corr_a;
}

double pair_data_olpslater1s1s_get_corr_b(pair_pot_type *pair_pot) {
  return (*(pair_data_olpslater1s1s_type*)((*pair_pot).pair_data)).corr_b;
}

double pair_data_olpslater1s1s_get_corr_c(pair_pot_type *pair_pot) {
  return (*(pair_data_olpslater1s1s_type*)((*pair_pot).pair_data)).corr_c;
}


void pair_data_chargetransferslater1s1s_init(pair_pot_type *pair_pot, double *slater1s_widths, double *slater1s_N, double ct_scale, double width_power) {
  pair_data_chargetransferslater1s1s_type *pair_data;
  pair_data = malloc(sizeof(pair_data_chargetransferslater1s1s_type));
  (*pair_pot).pair_data = pair_data;
  if (pair_data != NULL) {
    (*pair_pot).pair_fn = pair_fn_chargetransferslater1s1s;
    (*pair_data).widths = slater1s_widths;
    (*pair_data).N = slater1s_N;
    (*pair_data).ct_scale = ct_scale;
    (*pair_data).width_power = width_power;
  }
}


double pair_fn_chargetransferslater1s1s(void *pair_data, long center_index, long other_index, double d, double *delta, double *g, double *g_cart) {
  double a, b, Na, Nb, fac;
  double pot = 0.0;
  a  = (*(pair_data_olpslater1s1s_type*)pair_data).widths[center_index];
  b  = (*(pair_data_olpslater1s1s_type*)pair_data).widths[other_index];
  Na = (*(pair_data_olpslater1s1s_type*)pair_data).N[center_index];
  Nb = (*(pair_data_olpslater1s1s_type*)pair_data).N[other_index];
  // Overlap between unit Slater densities
  pot += slater_00_00_olp(a, b, d, g);
  // Multiply with scaling factor and populations
  pot *= -Na*Nb*(*(pair_data_chargetransferslater1s1s_type*)pair_data).ct_scale;
  if (g != NULL) *g *= -Na*Nb*(*(pair_data_chargetransferslater1s1s_type*)pair_data).ct_scale;
  // Multiply with power of widths
  // TODO: this power will likely be integer, so this could be implemented more efficiently
  double wp = (*(pair_data_chargetransferslater1s1s_type*)pair_data).width_power;
  if (wp == 3.0) {
    fac = 1.0/a/b;
    fac *= fac*fac;
  } else {
    fac = pow( 1.0/a/b, wp);
  }
  pot *= fac;
  if (g != NULL) *g *= fac;
  return pot;
}

double pair_data_chargetransferslater1s1s_get_ct_scale(pair_pot_type *pair_pot) {
  return (*(pair_data_chargetransferslater1s1s_type*)((*pair_pot).pair_data)).ct_scale;
}

double pair_data_chargetransferslater1s1s_get_width_power(pair_pot_type *pair_pot) {
  return (*(pair_data_chargetransferslater1s1s_type*)((*pair_pot).pair_data)).width_power;
}
