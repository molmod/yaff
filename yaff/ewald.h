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


#ifndef YAFF_EWALD_H
#define YAFF_EWALD_H

#include "pair_pot.h"

#define M_TWO_PI 6.2831853071795864769
#define M_SQRT_PI 1.7724538509055160273


double compute_ewald_reci(double *pos, long natom, double *charges,
                          double *gvecs, double volume, double alpha,
                          long *gmax, double *gradient, double *work);
double compute_ewald_corr(double *pos, long center_index, double *charges,
                          double *rvecs, double *gvecs, double alpha, 
                          scaling_row_type *scaling, long scaling_size,
                          double *gradient);

#endif
