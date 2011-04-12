# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


cimport numpy as np
cimport nlists

cdef extern from "pair_pot.h":
    ctypedef struct scaling_row_type:
        np.long_t i
        np.float64_t scale
    
    ctypedef struct pair_pot_type:
        pass

    pair_pot_type* pair_pot_new()
    void pair_pot_free(pair_pot_type *pair_pot)
    bint pair_pot_ready(pair_pot_type *pair_pot)
    double pair_pot_get_cutoff(pair_pot_type *pair_pot)
    void pair_pot_set_cutoff(pair_pot_type *pair_pot, double cutoff)
    void pair_data_free(pair_pot_type *pair_pot)
    
    double pair_pot_energy(long center_index, nlists.nlist_row_type* nlist,
                           long nlist_size, scaling_row_type* scaling,
                           long scaling_size, pair_pot_type* pair_pot)

    void pair_data_lj_init(pair_pot_type *pair_pot, double *sigma, double *epsilon)
    void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha)

