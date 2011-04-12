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

#include "mic.h"
#include <math.h>

void mic(double *delta, double *rvecs, double *gvecs, long nvec) {
  // applies the Minimum Image Convention.
  long i;
  double x;
  for (i=0; i<nvec; i++) {
    x = gvecs[3*i]*delta[0] + gvecs[3*i+1]*delta[1] + gvecs[3*i+2]*delta[2];
    x = ceil(x-0.5);
    delta[0] -= x*rvecs[3*i];
    delta[1] -= x*rvecs[3*i+1];
    delta[2] -= x*rvecs[3*i+2];
  }
}
