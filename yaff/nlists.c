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

#include "nlists.h"
#include <math.h>


int nlist_update_low(double *pos, long center_index, double cutoff, long *rmax,
                     double *rvecs, double *gvecs, long *nlist_status,
                     nlist_row_type *nlist, long pos_size, long nlist_size,
                     int nvec) {

  long other_index, row;
  long *r;
  int update_delta0, i;
  double delta0[3], delta[3], x;
  double *center_pos;

  r = nlist_status;
  other_index = nlist_status[3];
  center_pos = pos + 3*center_index;

  update_delta0 = 1;
  row = 0;

  while (row < nlist_size) {
    if (other_index >= pos_size) {
      nlist_status[4] += row;
      return 1;
    }
    if (update_delta0) {
      // Compute the relative vector.
      delta0[0] = center_pos[0] - pos[3*other_index];
      delta0[1] = center_pos[1] - pos[3*other_index+1];
      delta0[2] = center_pos[2] - pos[3*other_index+2];
      // Subtract the cell vectors as to make the relative vector as short
      // as possible. (This is the minimum image convention.)
      for (i=0; i<nvec; i++) {
        x = gvecs[3*i]*delta0[0] + gvecs[3*i+1]*delta0[1] + gvecs[3*i+2]*delta0[2];
        x = ceil(x-0.5);
        delta0[0] -= x*rvecs[3*i];
        delta0[1] -= x*rvecs[3*i+1];
        delta0[2] -= x*rvecs[3*i+2];
      }
      // Done updating delta0.
      update_delta0 = 0;
    }
    // Construct delta by adding the appropriate cell vector to delta0
    delta[0] = delta0[0];
    delta[1] = delta0[1];
    delta[2] = delta0[2];
    for (i=0; i<nvec; i++) {
      delta[0] += r[i]*rvecs[3*i];
      delta[1] += r[i]*rvecs[3*i+1];
      delta[2] += r[i]*rvecs[3*i+2];
    }
    // Compute the distance and store the record if distance is below the cutoff.
    x = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    if (x < cutoff) {
      (*nlist).i = other_index;
      (*nlist).d = x;
      (*nlist).dx = delta[0];
      (*nlist).dy = delta[1];
      (*nlist).dz = delta[2];
      (*nlist).r0 = r[0];
      (*nlist).r1 = r[1];
      (*nlist).r2 = r[2];
      nlist++;
      row++;
    }
    // Increase the appropriate counters in the quadruple loop.
    if (nvec > 0) {
      r[0]++;
      if (r[0] > rmax[0]) {
        r[0] = -rmax[0];
        if (nvec > 1) {
          r[1]++;
          if (r[1] > rmax[1]) {
            r[1] = -rmax[1];
            if (nvec > 2) {
              r[2]++;
              if (r[2] > rmax[2]) {
                r[2] = -rmax[2];
                other_index++;
                update_delta0 = 1;
              }
            } else {
              other_index++;
              update_delta0 = 1;
            }
          }
        } else {
          other_index++;
          update_delta0 = 1;
        }
      }
    } else {
      other_index++;
      update_delta0 = 1;
    }
  }
  // Exit before the job is done. Keep track of the status. Work will be resumed
  // in a next call.
  nlist_status[0] = r[0];
  nlist_status[1] = r[1];
  nlist_status[2] = r[2];
  nlist_status[3] = other_index;
  nlist_status[4] += row;
  return 0;
}
