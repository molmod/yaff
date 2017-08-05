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
#include "dlist.h"
#include "cell.h"

void dlist_forward(double *pos, cell_type *unitcell, dlist_row_type* deltas, long ndelta) {
  long k;
  dlist_row_type *delta;
  for (k=0; k<ndelta; k++) {
    delta = (deltas + k);
    (*delta).dx = pos[3*(*delta).j    ] - pos[3*(*delta).i    ];
    (*delta).dy = pos[3*(*delta).j + 1] - pos[3*(*delta).i + 1];
    (*delta).dz = pos[3*(*delta).j + 2] - pos[3*(*delta).i + 2];
    cell_mic((double*)delta, unitcell);
    (*delta).gx = 0.0;
    (*delta).gy = 0.0;
    (*delta).gz = 0.0;
  }
}

void dlist_back(double *gpos, double *vtens, dlist_row_type* deltas, long ndelta) {
  long k;
  dlist_row_type *delta;
  for (k=0; k<ndelta; k++) {
    delta = (deltas + k);
    if (gpos != NULL) {
      gpos[3*(*delta).j    ] += (*delta).gx;
      gpos[3*(*delta).j + 1] += (*delta).gy;
      gpos[3*(*delta).j + 2] += (*delta).gz;
      gpos[3*(*delta).i    ] -= (*delta).gx;
      gpos[3*(*delta).i + 1] -= (*delta).gy;
      gpos[3*(*delta).i + 2] -= (*delta).gz;
    }
    if (vtens != NULL) {
      vtens[0] += (*delta).gx*(*delta).dx;
      vtens[1] += (*delta).gy*(*delta).dx;
      vtens[2] += (*delta).gz*(*delta).dx;
      vtens[3] += (*delta).gx*(*delta).dy;
      vtens[4] += (*delta).gy*(*delta).dy;
      vtens[5] += (*delta).gz*(*delta).dy;
      vtens[6] += (*delta).gx*(*delta).dz;
      vtens[7] += (*delta).gy*(*delta).dz;
      vtens[8] += (*delta).gz*(*delta).dz;
    }
  }
}
