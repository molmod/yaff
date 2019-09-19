# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
# --


from __future__ import division
from __future__ import print_function

import tempfile
import shutil
import os
import numpy as np
import pkg_resources

from molmod.test.common import tmpdir
from yaff.external.lammpsio import *
from yaff import System

from yaff.test.common import get_system_water32

def test_lammps_system_data_water32():
    system = get_system_water32()
    with tmpdir(__name__, 'test_lammps_system_water32') as dirname:
        fn = os.path.join(dirname,'lammps.system')
        write_lammps_system_data(system,fn=fn)
        with open(fn,'r') as f: lines = f.readlines()
        natom = int(lines[2].split()[0])
        assert natom==system.natom
        assert (system.natom+system.bonds.shape[0]+23)==len(lines)


def test_lammps_ffconversion_mil53():
    fn_system = pkg_resources.resource_filename(__name__, '../../data/test/system_mil53.chk')
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_mil53.txt')
    system = System.from_file(fn_system)
    with tmpdir(__name__, 'test_lammps_ffconversion_mil53') as dirname:
        ff2lammps(system, fn_pars, dirname)
        # No test for correctness, just check that output files are present
        assert os.path.isfile(os.path.join(dirname,'lammps.in'))
        assert os.path.isfile(os.path.join(dirname,'lammps.data'))
