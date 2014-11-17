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


#ifndef YAFF_EWALD_H
#define YAFF_EWALD_H

#include "pair_pot.h"
#include "cell.h"

double compute_ewald_reci(double *pos, long natom, double *charges,
                          cell_type* unitcell, double alpha, long *gmax, double
                          gcut, double dielectric, double *gpos, double *work,
                          double* vtens);
double compute_ewald_reci_dd(double *pos, long natom, double *charges, double *dipoles,
                          cell_type* unitcell, double alpha, long *gmax,
                          double gcut, double *gpos, double *work,
                          double* vtens);
double compute_ewald_corr(double *pos, double *charges,
                          cell_type *unitcell, double alpha,
                          scaling_row_type *stab, long stab_size,
                          double dielectric, double *gpos, double *vtens,
                          long natom);
double compute_ewald_corr_dd(double *pos, double *charges, double *dipoles,
                          cell_type *unitcell, double alpha,
                          scaling_row_type *stab, long stab_size,
                          double *gpos, double *vtens, long natom);
#endif
