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
      (*cell).volume = fabs(
        rvecs[0]*(rvecs[4]*rvecs[8]-rvecs[5]*rvecs[7])+
        rvecs[1]*(rvecs[5]*rvecs[6]-rvecs[3]*rvecs[8])+
        rvecs[2]*(rvecs[3]*rvecs[7]-rvecs[4]*rvecs[6])
      );
      break;
  }
}

void cell_mic(double *delta, cell_type* cell) {
  // Applies the Minimum Image Convention. Well, sort of. It does not always work like this.
  // This function contains an unrolled loop for speed.
  int nvec;
  double x;
  double *rvecs;
  double *gvecs;
  nvec = (*cell).nvec;
  if (nvec == 0) return;
  rvecs = (*cell).rvecs;
  gvecs = (*cell).gvecs;
  x = ceil(gvecs[0]*delta[0] + gvecs[1]*delta[1] + gvecs[2]*delta[2] - 0.5);
  delta[0] -= x*rvecs[0];
  delta[1] -= x*rvecs[1];
  delta[2] -= x*rvecs[2];
  if (nvec == 1) return;
  x = ceil(gvecs[3]*delta[0] + gvecs[4]*delta[1] + gvecs[5]*delta[2] - 0.5);
  delta[0] -= x*rvecs[3];
  delta[1] -= x*rvecs[4];
  delta[2] -= x*rvecs[5];
  if (nvec == 2) return;
  x = ceil(gvecs[6]*delta[0] + gvecs[7]*delta[1] + gvecs[8]*delta[2] - 0.5);
  delta[0] -= x*rvecs[6];
  delta[1] -= x*rvecs[7];
  delta[2] -= x*rvecs[8];
}


void cell_to_center(double *cart, cell_type* cell, long *center) {
  // Transfroms to fractional coordinates
  int nvec;
  double *gvecs;
  nvec = (*cell).nvec;
  if (nvec == 0) return;
  gvecs = (*cell).gvecs;
  center[0] = -ceil(gvecs[0]*cart[0] + gvecs[1]*cart[1] + gvecs[2]*cart[2] - 0.5);
  if (nvec == 1) return;
  center[1] = -ceil(gvecs[3]*cart[0] + gvecs[4]*cart[1] + gvecs[5]*cart[2] - 0.5);
  if (nvec == 2) return;
  center[2] = -ceil(gvecs[6]*cart[0] + gvecs[7]*cart[1] + gvecs[8]*cart[2] - 0.5);
}


void cell_add_vec(double *delta, cell_type* cell, long* r) {
  // Simply adds an linear combination of cell vectors to delta.
  // This function contains an unrolled loop for speed.
  int nvec;
  double *rvecs;
  nvec = (*cell).nvec;
  if (nvec == 0) return;
  rvecs = (*cell).rvecs;
  delta[0] += r[0]*rvecs[0];
  delta[1] += r[0]*rvecs[1];
  delta[2] += r[0]*rvecs[2];
  if (nvec == 1) return;
  delta[0] += r[1]*rvecs[3];
  delta[1] += r[1]*rvecs[4];
  delta[2] += r[1]*rvecs[5];
  if (nvec == 2) return;
  delta[0] += r[2]*rvecs[6];
  delta[1] += r[2]*rvecs[7];
  delta[2] += r[2]*rvecs[8];
}


int is_invalid_exclude(long* exclude, long natom0, long natom1, long nexclude, int intra) {
  long iex;
  for (iex = 0; iex < nexclude; iex++) {
    if (exclude[2*iex] < 0) return 1;
    if (exclude[2*iex] >= natom0) return 1;
    if (exclude[2*iex+1] < 0) return 1;
    if (exclude[2*iex+1] >= natom1) return 1;
    if (intra) {
      if (exclude[2*iex+1] >= exclude[2*iex]) return 1;
    }
    if (iex > 0) {
      if (exclude[2*iex] < exclude[2*iex-2]) return 1;
      if (exclude[2*iex] == exclude[2*iex-2]) {
        if (exclude[2*iex+1] <= exclude[2*iex-1]) return 1;
      }
    }
  }
  return 0;
}


int is_excluded(long* exclude, long i0, long i1, long* iex, long nexclude) {
  if (*iex >= nexclude) return 0;
  while (exclude[2*(*iex)] < i0) {
    (*iex)++;
    if (*iex >= nexclude) return 0;
  }
  if (exclude[2*(*iex)] != i0) return 0;
  while (exclude[2*(*iex)+1] < i1) {
    (*iex)++;
    if (*iex >= nexclude) return 0;
    if (exclude[2*(*iex)] != i0) return 0;
  }
  return ((exclude[2*(*iex)+1] == i1) && (exclude[2*(*iex)] == i0));
}


double helper_distances(double* pos0, double* pos1, cell_type* cell, long* r) {
  double delta[3];
  delta[0] = pos0[0] - pos1[0];
  delta[1] = pos0[1] - pos1[1];
  delta[2] = pos0[2] - pos1[2];
  cell_mic(delta, cell);
  if (r != NULL) {
    cell_add_vec(delta, cell, r);
  }
  return sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
}

void cell_compute_distances1(cell_type* cell, double* pos, double* output, long natom, long* pairs, long npair, int do_include, long nimage) {
  if (do_include) {
    long ipair;
    for (ipair=0; ipair<npair; ipair++) {
      *output = helper_distances(pos + 3*pairs[2*ipair], pos + 3*pairs[2*ipair+1], cell, NULL);
      output++;
    }
  } else {
    long r[3];
    long i0, i1, ipair, r0, r1, r2;
    ipair = 0;
    for (r0=-nimage; r0 <= nimage; r0++) {
      r[0] = r0;
      for (r1=-nimage; r1 <= nimage; r1++) {
        r[1] = r1;
        for (r2=-nimage; r2 <= nimage; r2++) {
          r[2] = r2;
          for (i0=0; i0<natom; i0++) {
            for (i1=0; i1<i0; i1++) {
              if ((r0==0) && (r1==0) && (r2==0)) {
                if (is_excluded(pairs, i0, i1, &ipair, npair)) continue;
              }
              *output = helper_distances(pos + 3*i0, pos + 3*i1, cell, r);
              output++;
            }
          }
        }
      }
    }
  }
}

void cell_compute_distances2(cell_type* cell, double* pos0, double* pos1, double* output, long natom0, long natom1, long* pairs, long npair, int do_include, long nimage) {
  if (do_include) {
    long ipair;
    for (ipair=0; ipair<npair; ipair++) {
      *output = helper_distances(pos0 + 3*pairs[2*ipair], pos1 + 3*pairs[2*ipair+1], cell, NULL);
      output++;
    }
  } else {
    long r[3];
    long i0, i1, ipair, r0, r1, r2;
    ipair = 0;
    for (r0=-nimage; r0 <= nimage; r0++) {
      r[0] = r0;
      for (r1=-nimage; r1 <= nimage; r1++) {
        r[1] = r1;
        for (r2=-nimage; r2 <= nimage; r2++) {
          r[2] = r2;
          for (i0=0; i0<natom0; i0++) {
            for (i1=0; i1<natom1; i1++) {
              if ((r0==0) && (r1==0) && (r2==0)) {
                if (is_excluded(pairs, i0, i1, &ipair, npair)) continue;
              }
              *output = helper_distances(pos0 + 3*i0, pos1 + 3*i1, cell, r);
              output++;
            }
          }
        }
      }
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

void cell_to_frac(cell_type* cell, double *cart, double* frac) {
  double *gvecs = (*cell).gvecs;
  frac[0] = gvecs[0]*cart[0] + gvecs[1]*cart[1] + gvecs[2]*cart[2];
  frac[1] = gvecs[3]*cart[0] + gvecs[4]*cart[1] + gvecs[5]*cart[2];
  frac[2] = gvecs[6]*cart[0] + gvecs[7]*cart[1] + gvecs[8]*cart[2];
}
