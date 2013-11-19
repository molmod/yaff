# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
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
#--


import numpy as np

from molmod import kcalmol, angstrom, rad, deg, femtosecond, boltzmann
from molmod.periodic import periodic
from molmod.io import XYZWriter

from yaff import *

from yaff.test.common import get_system_water
from yaff.pes.test.common import check_gpos_ff, check_vtens_ff
from yaff.pes.test.test_pair_pot import get_part_water_eidip
from yaff.sampling.polarization import RelaxDipoles


def test1():
    system, nlist, scalings, part_pair, pair_pot, pair_fn = get_part_water_eidip()
    ff = ForceField(system, [part_pair], nlist)
    poltens = np.tile( np.diag([1,1,1]) , np.array([system.natom, 1]) )

    opt = CGOptimizer(CartesianDOF(ff), hooks=RelaxDipoles(poltens))
    opt.run(2)
