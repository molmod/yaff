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
#include "ewald.h"
#include "cell.h"


double compute_ewald_reci(double *pos, long natom, double *charges,
                          cell_type* cell, double alpha, long *gmax,
                          double *gpos, double *work, double* vtens) {
  long j0, j1, j2, i;
  double energy, k[3], ksq, cosfac, sinfac, x, c, s, fac1, fac2;
  double kvecs[9];
  for (i=0; i<9; i++) {
    kvecs[i] = M_TWO_PI*(*cell).gvecs[i];
  }
  energy = 0.0;
  fac1 = M_FOUR_PI/(*cell).volume;
  fac2 = 0.25/alpha/alpha;
  for (j0=-gmax[0]; j0 <= gmax[0]; j0++) {
    for (j1=-gmax[1]; j1 <= gmax[1]; j1++) {
      for (j2=0; j2 <= gmax[2]; j2++) {
        if (j2==0) {
          if (j1<0) continue;
          if ((j1==0)&&(j0<=0)) continue;
        }
        k[0] = (j0*kvecs[0] + j1*kvecs[3] + j2*kvecs[6]);
        k[1] = (j0*kvecs[1] + j1*kvecs[4] + j2*kvecs[7]);
        k[2] = (j0*kvecs[2] + j1*kvecs[5] + j2*kvecs[8]);
        ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
        cosfac = 0.0;
        sinfac = 0.0;
        for (i=0; i<natom; i++) {
          x = k[0]*pos[3*i] + k[1]*pos[3*i+1] + k[2]*pos[3*i+2];
          c = charges[i]*cos(x);
          s = charges[i]*sin(x);
          cosfac += c;
          sinfac += s;
          if (gpos != NULL) {
            work[2*i] = c;
            work[2*i+1] = -s;
          }
        }
        c = fac1*exp(-ksq*fac2)/ksq;
        s = (cosfac*cosfac+sinfac*sinfac);
        energy += c*s;
        if (gpos != NULL) {
          for (i=0; i<natom; i++) {
            x = 2.0*c*(cosfac*work[2*i+1] + sinfac*work[2*i]);
            gpos[3*i] += k[0]*x;
            gpos[3*i+1] += k[1]*x;
            gpos[3*i+2] += k[2]*x;
          }
        }
        if (vtens != NULL) {
          c *= 2.0*(1.0/ksq+fac2)*s;
          vtens[0] += c*k[0]*k[0];
          vtens[4] += c*k[1]*k[1];
          vtens[8] += c*k[2]*k[2];
          x = c*k[1]*k[0];
          vtens[1] += x;
          vtens[3] += x;
          x = c*k[2]*k[0];
          vtens[2] += x;
          vtens[6] += x;
          x = c*k[2]*k[1];
          vtens[5] += x;
          vtens[7] += x;
        }
      }
    }
  }
  if (vtens != NULL) {
    vtens[0] -= energy;
    vtens[4] -= energy;
    vtens[8] -= energy;
  }
  return energy;
}

double compute_ewald_corr(double *pos, long center_index, double *charges,
                          cell_type *unitcell, double alpha,
                          scaling_row_type *scaling, long scaling_size,
                          double *gpos, double *vtens) {
  long i, other_index;
  double energy, delta[3], d, x, g, pot, fac;
  energy = 0.0;
  // Self-interaction correction (no gpos or vtens contribution)
  energy -= alpha/M_SQRT_PI*charges[center_index]*charges[center_index];
  // Scaling corrections
  for (i = 0; i < scaling_size; i++) {
    other_index = scaling[i].i;
    if (other_index >= center_index) continue; // avoid double counting.
    delta[0] = pos[3*center_index    ] - pos[3*other_index    ];
    delta[1] = pos[3*center_index + 1] - pos[3*other_index + 1];
    delta[2] = pos[3*center_index + 2] - pos[3*other_index + 2];
    cell_mic(delta, unitcell);
    d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    x = alpha*d;
    pot = erf(x)/d;
    fac = (1-scaling[i].scale)*charges[other_index]*charges[center_index];
    if ((gpos != NULL) || (vtens != NULL)) {
      g = -fac*(M_TWO_DIV_SQRT_PI*alpha*exp(-x*x) - pot)/d/d;
    }
    if (gpos != NULL) {
      x = delta[0]*g;
      gpos[3*center_index  ] += x;
      gpos[3*other_index   ] -= x;
      x = delta[1]*g;
      gpos[3*center_index+1] += x;
      gpos[3*other_index +1] -= x;
      x = delta[2]*g;
      gpos[3*center_index+2] += x;
      gpos[3*other_index +2] -= x;
    }
    if (vtens != NULL) {
      vtens[0] += delta[0]*delta[0]*g;
      vtens[4] += delta[1]*delta[1]*g;
      vtens[8] += delta[2]*delta[2]*g;
      x = delta[1]*delta[0]*g;
      vtens[1] += x;
      vtens[3] += x;
      x = delta[2]*delta[0]*g;
      vtens[2] += x;
      vtens[6] += x;
      x = delta[2]*delta[1]*g;
      vtens[5] += x;
      vtens[7] += x;
    }
    energy -= fac*pot;
  }
  return energy;
}
