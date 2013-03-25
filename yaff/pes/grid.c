// YAFF is yet another force-field code
// Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
//--


//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

#include "grid.h"


double compute_grid3d(double* center, cell_type *cell, double* egrid, long* shape) {
    double frac[3], e11, e12, e21, e22;
    long indexes[3];
    int i;

    cell_to_frac(cell, center, frac);
#ifdef DEBUG
    printf("frac = [%f, %f, %f]\n", frac[0], frac[1], frac[2]);
#endif

    for (i=0; i<3; i++) {
        // Move to ranges [0,1[
        frac[i] -= floor(frac[i]);

        // Convert to grid indexes
        frac[i] *= shape[i];
        indexes[i] = (long)floor(frac[i]);
        frac[i] -= indexes[i];
    }

#define offset(inc0,inc1,inc2)  ((((indexes[0]+inc0)%shape[0])*shape[1] + (indexes[1]+inc1)%shape[1])*shape[2] + (indexes[2]+inc2)%shape[2])

#ifdef DEBUG
    printf("center = [%f, %f, %f]\n", center[0], center[1], center[2]);
    printf("frac = [%f, %f, %f]\n", frac[0], frac[1], frac[2]);
    printf("indexes = [%li, %li, %li]\n", indexes[0], indexes[1], indexes[2]);
    printf("\n");
#endif

    // trilinear interpolation
    return
    /* 111 */ ((egrid[offset(1,1,1)]*frac[0] +
    /* 011 */   egrid[offset(0,1,1)]*(1-frac[0]))*frac[1] +
    /* 101 */  (egrid[offset(1,0,1)]*frac[0] +
    /* 001 */   egrid[offset(0,0,1)]*(1-frac[0]))*(1-frac[1]))*frac[2] +
    /* 110 */ ((egrid[offset(1,1,0)]*frac[0] +
    /* 010 */   egrid[offset(0,1,0)]*(1-frac[0]))*frac[1] +
    /* 100 */  (egrid[offset(1,0,0)]*frac[0] +
    /* 000 */   egrid[offset(0,0,0)]*(1-frac[0]))*(1-frac[1]))*(1-frac[2]);
}
