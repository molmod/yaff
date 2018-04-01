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


#include "comlist.h"


void comlist_forward(dlist_row_type* deltas, double *pos, double* compos, long* comsizes,
    comlist_row_type* comtab, long ncom) {
  int icom;
  for (icom = 0; icom < ncom; icom++) {
    double cx = 0, cy = 0, cz = 0;
    long i0 = (*comtab).i;
    double wtot = (*comtab).w;
    int i;
    comtab++;
    for (i = *comsizes - 2; i >= 0; i--) {
      cx += (*comtab).w*(*deltas).dx;
      cy += (*comtab).w*(*deltas).dy;
      cz += (*comtab).w*(*deltas).dz;
      deltas++;
      comtab++;
    }
    comsizes++;
    compos[icom*3] = cx/wtot + pos[i0*3];
    compos[icom*3 + 1] = cy/wtot + pos[i0*3 + 1];
    compos[icom*3 + 2] = cz/wtot + pos[i0*3 + 2];
  }
}

void comlist_back(dlist_row_type* deltas, double *gpos, double* gcompos, long* comsizes,
    comlist_row_type* comtab, long ncom) {
  int icom;
  for (icom = 0; icom < ncom; icom++) {
    long i0 = (*comtab).i;
    double wtot = (*comtab).w;
    double gcx = gcompos[3*icom];
    double gcy = gcompos[3*icom + 1];
    double gcz = gcompos[3*icom + 2];
    int i;
    comtab++;
    gpos[3*i0] += gcx;
    gpos[3*i0 + 1] += gcy;
    gpos[3*i0 + 2] += gcz;
    for (i = *comsizes - 2; i >= 0; i--) {
      double wratio = (*comtab).w/wtot;
      (*deltas).gx += gcx*wratio;
      (*deltas).gy += gcy*wratio;
      (*deltas).gz += gcz*wratio;
      deltas++;
      comtab++;
    }
    comsizes++;
  }
}
