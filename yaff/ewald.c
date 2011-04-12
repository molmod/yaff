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
#include "ewald.h"
#include "mic.h"


double compute_ewald_reci(double *pos, long natom, double *charges,
                          double *gvecs, double volume, double alpha,
                          long *gmax, double *gradient) {
  long j0, j1, j2, i;
  double energy, k[3], ksq, cosfac, sinfac, x;
  energy = 0.0;
  for (j0=-gmax[0]; j0 <= gmax[0]; j0++) {
    for (j1=-gmax[1]; j1 <= gmax[1]; j1++) {
      for (j2=-gmax[2]; j2 <= gmax[2]; j2++) {
        if ((j0==0)&&(j1==0)&(j2==0)) continue;
        k[0] = M_TWO_PI*(j0*gvecs[0] + j1*gvecs[3] + j2*gvecs[6]);
        k[1] = M_TWO_PI*(j0*gvecs[1] + j1*gvecs[4] + j2*gvecs[7]);
        k[2] = M_TWO_PI*(j0*gvecs[2] + j1*gvecs[5] + j2*gvecs[8]);
        ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
        cosfac = 0.0;
        sinfac = 0.0;
        for (i=0; i<natom; i++) {
          x = k[0]*pos[3*i] + k[1]*pos[3*i+1] + k[2]*pos[3*i+2];
          cosfac += charges[i]*cos(x);
          sinfac += charges[i]*sin(x);
        }
        energy += exp(-ksq/4.0/alpha/alpha)/ksq*(cosfac*cosfac+sinfac*sinfac);
      }
    }
  }
  energy *= M_TWO_PI/volume;
  return energy;
}

double compute_ewald_corr(double *pos, long center_index, double *charges,
                          double *rvecs, double *gvecs, double alpha,
                          scaling_row_type *scaling, long scaling_size,
                          double *gradient) {
  long i, other_index;
  double energy, delta[3], d, s;
  energy = 0.0;
  // Self-interaction correction
  energy -= alpha/M_SQRT_PI*charges[center_index]*charges[center_index];
  // Scaling corrections
  for (i = 0; i < scaling_size; i++) {
    other_index = scaling[i].i;
    s = scaling[i].scale;
    if (other_index >= center_index) continue; // avoid double counting.
    delta[0] = pos[3*center_index    ] - pos[3*other_index    ];
    delta[1] = pos[3*center_index + 1] - pos[3*other_index + 1];
    delta[2] = pos[3*center_index + 2] - pos[3*other_index + 2];
    mic(delta, rvecs, gvecs, 3);
    d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    energy -= (1-s)*charges[other_index]*charges[center_index]*erf(alpha*d)/d;
  }
  return energy;
}
