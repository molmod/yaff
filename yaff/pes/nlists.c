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
#include "nlists.h"
#include "cell.h"


int nlist_update_low(double *pos, long center_index, double rcut, long *rmax,
                     cell_type *unitcell, long *nlist_status,
                     nlist_row_type *nlist, long pos_size, long nlist_size) {

  long other_index, row;
  long *r;
  int update_delta0, i;
  double delta0[3], delta[3], d;
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
      delta0[0] = pos[3*other_index  ] - center_pos[0];
      delta0[1] = pos[3*other_index+1] - center_pos[1];
      delta0[2] = pos[3*other_index+2] - center_pos[2];
      // Subtract the cell vectors as to make the relative vector as short
      // as possible. (This is the minimum image convention.)
      cell_mic(delta0, unitcell);
      // Done updating delta0.
      update_delta0 = 0;
    }
    // Construct delta by adding the appropriate cell vector to delta0
    delta[0] = delta0[0];
    delta[1] = delta0[1];
    delta[2] = delta0[2];
    for (i=0; i<(*unitcell).nvec; i++) {
      delta[0] += r[i]*(*unitcell).rvecs[3*i];
      delta[1] += r[i]*(*unitcell).rvecs[3*i+1];
      delta[2] += r[i]*(*unitcell).rvecs[3*i+2];
    }
    // Compute the distance and store the record if distance is below the rcut.
    d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    if (d < rcut) {
      if ((r[0]!=0)||(r[1]!=0)||(r[2]!=0)||(center_index<other_index)) {
        (*nlist).i = other_index;
        (*nlist).d = d;
        (*nlist).dx = delta[0];
        (*nlist).dy = delta[1];
        (*nlist).dz = delta[2];
        (*nlist).r0 = r[0];
        (*nlist).r1 = r[1];
        (*nlist).r2 = r[2];
        nlist++;
        row++;
      }
    }
    // Increase the appropriate counters in the quadruple loop.
    if (!inc_r(unitcell, r, rmax)) {
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


int inc_r(cell_type *unitcell, long *r, long *rmax) {
  // increment the counters for the periodic images.
  // returns true when the counters were incremented succesfully.
  // returns false and resets all the counters when the iteration over all cells
  // is complete.
  if ((*unitcell).nvec > 0) {
    r[0]++;
    if (r[0] > rmax[0]) {
      r[0] = -rmax[0];
      if ((*unitcell).nvec > 1) {
        r[1]++;
        if (r[1] > rmax[1]) {
          r[1] = -rmax[1];
          if ((*unitcell).nvec > 2) {
            r[2]++;
            if (r[2] > rmax[2]) {
              r[2] = -rmax[2];
              return 0;
            }
          } else {
            return 0;
          }
        }
      } else {
        return 0;
      }
    }
  } else {
    return 0;
  }
  return 1;
}
