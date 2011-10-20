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
#include "nlist.h"
#include "cell.h"


int nlist_update_low(double *pos, double rcut, long *rmax,
                     cell_type *unitcell, long *status,
                     neigh_row_type *neighs, long natom, long nneigh) {

  long a, b, row;
  long *r;
  int update_delta0, i;
  double delta0[3], delta[3], d;

  r = status;
  a = status[3];
  b = status[4];

  update_delta0 = 1;
  row = 0;

  while (row < nneigh) {
    if (a >= natom) {
      // Completely done.
      status[5] += row;
      return 1;
    }
    if (update_delta0) {
      // Compute the relative vector.
      delta0[0] = pos[3*b  ] - pos[3*a  ];
      delta0[1] = pos[3*b+1] - pos[3*a+1];
      delta0[2] = pos[3*b+2] - pos[3*a+2];
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
    if ((r[0]!=0)||(r[1]!=0)||(r[2]!=0)||(a>b)) {
      // Compute the distance and store the record if distance is below the rcut.
      d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
      if (d < rcut) {
        // it is crucial that a<b for the central cell!!! See pair_pot.c:get_scaling
        (*neighs).a = a;
        (*neighs).b = b;
        (*neighs).d = d;
        (*neighs).dx = delta[0];
        (*neighs).dy = delta[1];
        (*neighs).dz = delta[2];
        (*neighs).r0 = r[0];
        (*neighs).r1 = r[1];
        (*neighs).r2 = r[2];
        neighs++;
        row++;
      }
    }
    // Increase the appropriate counters in the quintuple loop.
    if (!nlist_inc_r(unitcell, r, rmax)) {
      update_delta0 = 1;
      b++;
      if (b >= natom) {
        b = 0;
        a++;
      }
    }
  }
  // Exit before the job is done. Keep track of the status. Work will be resumed
  // in a next call.
  status[0] = r[0];
  status[1] = r[1];
  status[2] = r[2];
  status[3] = a;
  status[4] = b;
  status[5] += row;
  return 0;
}


int nlist_inc_r(cell_type *unitcell, long *r, long *rmax) {
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
