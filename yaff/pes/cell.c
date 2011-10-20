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


#include <stdlib.h>
#include <math.h>
#include "cell.h"

cell_type* cell_new(void) {
  return malloc(sizeof(cell_type));
}

void cell_free(cell_type* cell) {
  free(cell);
}

void cell_update(cell_type* cell, double *rvecs, double *gvecs, int nvec) {
  double tmp;
  int i;
  // just copy everything
  (*cell).nvec = nvec;
  for (i=0; i<9; i++) {
    (*cell).rvecs[i] = rvecs[i];
    (*cell).gvecs[i] = gvecs[i];
  }
  // compute the spacings
  for (i=0; i<3; i++) {
    (*cell).rspacings[i] = 1.0/sqrt(gvecs[3*i]*gvecs[3*i] + gvecs[3*i+1]*gvecs[3*i+1] + gvecs[3*i+2]*gvecs[3*i+2]);
    (*cell).gspacings[i] = 1.0/sqrt(rvecs[3*i]*rvecs[3*i] + rvecs[3*i+1]*rvecs[3*i+1] + rvecs[3*i+2]*rvecs[3*i+2]);
  }
  // compute the volume
  switch(nvec) {
    case 0:
      (*cell).volume = 0.0;
      break;
    case 1:
      (*cell).volume = sqrt(
        rvecs[0]*rvecs[0]+rvecs[1]*rvecs[1]+rvecs[2]*rvecs[2]
      );
      break;
    case 2:
      tmp = rvecs[0]*rvecs[3]+rvecs[1]*rvecs[4]+rvecs[2]*rvecs[5];
      tmp = (rvecs[0]*rvecs[0]+rvecs[1]*rvecs[1]+rvecs[2]*rvecs[2])*
            (rvecs[3]*rvecs[3]+rvecs[4]*rvecs[4]+rvecs[5]*rvecs[5]) - tmp*tmp;
      if (tmp > 0) {
        (*cell).volume = sqrt(tmp);
      } else {
        (*cell).volume = 0.0;
      }
      break;
    case 3:
      (*cell).volume = (
        rvecs[0]*(rvecs[4]*rvecs[8]-rvecs[5]*rvecs[7])+
        rvecs[1]*(rvecs[5]*rvecs[6]-rvecs[3]*rvecs[8])+
        rvecs[2]*(rvecs[3]*rvecs[7]-rvecs[4]*rvecs[6])
      );
      break;
  }
}

void cell_mic(double *delta, cell_type* cell) {
  // applies the Minimum Image Convention.
  long i;
  double x;
  for (i=0; i<(*cell).nvec; i++) {
    x = (*cell).gvecs[3*i]*delta[0] + (*cell).gvecs[3*i+1]*delta[1] + (*cell).gvecs[3*i+2]*delta[2];
    x = ceil(x-0.5);
    delta[0] -= x*(*cell).rvecs[3*i];
    delta[1] -= x*(*cell).rvecs[3*i+1];
    delta[2] -= x*(*cell).rvecs[3*i+2];
  }
}


void cell_add_vec(double *delta, cell_type* cell, long* r) {
  long i;
  for (i=0; i<(*cell).nvec; i++) {
    delta[0] += r[i]*(*cell).rvecs[3*i];
    delta[1] += r[i]*(*cell).rvecs[3*i+1];
    delta[2] += r[i]*(*cell).rvecs[3*i+2];
  }
}


void cell_compute_distances1(cell_type* cell, double* pos, double* output, long natom) {
  double delta[3];
  long i0, i1;
  for (i0=0; i0<natom; i0++) {
    for (i1=0; i1<i0; i1++) {
      delta[0] = pos[3*i0  ] - pos[3*i1  ];
      delta[1] = pos[3*i0+1] - pos[3*i1+1];
      delta[2] = pos[3*i0+2] - pos[3*i1+2];
      cell_mic(delta, cell);
      *output = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
      output++;
    }
  }
}

void cell_compute_distances2(cell_type* cell, double* pos0, double* pos1, double* output, long natom0, long natom1) {
  double delta[3];
  long i0, i1;
  for (i0=0; i0<natom0; i0++) {
    for (i1=0; i1<natom1; i1++) {
      delta[0] = pos0[3*i0  ] - pos1[3*i1  ];
      delta[1] = pos0[3*i0+1] - pos1[3*i1+1];
      delta[2] = pos0[3*i0+2] - pos1[3*i1+2];
      cell_mic(delta, cell);
      *output = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
      output++;
    }
  }
}

int cell_get_nvec(cell_type* cell) {
  return (*cell).nvec;
}

double cell_get_volume(cell_type* cell) {
  return (*cell).volume;
}

void cell_copy_rvecs(cell_type* cell, double *rvecs, int full) {
  int i, n;
  n = (full)?9:(*cell).nvec*3;
  for (i=0; i<n; i++) rvecs[i] = (*cell).rvecs[i];
}

void cell_copy_gvecs(cell_type* cell, double *gvecs, int full) {
  int i, n;
  n = (full)?9:(*cell).nvec*3;
  for (i=0; i<n; i++) gvecs[i] = (*cell).gvecs[i];
}

void cell_copy_rspacings(cell_type* cell, double *rspacings, int full) {
  int i, n;
  n = (full)?3:(*cell).nvec;
  for (i=0; i<n; i++) rspacings[i] = (*cell).rspacings[i];
}

void cell_copy_gspacings(cell_type* cell, double *gspacings, int full) {
  int i, n;
  n = (full)?3:(*cell).nvec;
  for (i=0; i<n; i++) gspacings[i] = (*cell).gspacings[i];
}
