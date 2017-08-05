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

import os
import shlex
import subprocess

from molmod.test.test_examples import check_example


def test_example_000_overview():
    check_example(__name__, '000_overview', 'simulation.py', ['parameters.txt', 'system.chk'])


def test_example_001_tutorial_bks():
    check_example(__name__, '001_tutorial_bks', 'all.sh', [
        'bks.pot', 'init/mksystem.py', 'init/rvecs.txt', 'init/struct.xyz',
        'nvt/analysis.py', 'nvt/simulation.py', 'opt/analysis.py', 'opt/simulation.py'])


def test_example_002_external_trajectory():
    check_example(__name__, '002_external_trajectory', 'rdf.py', ['trajectory.xyz'])


def test_example_003_water_thermostat():
    check_example(__name__, '003_water_thermostat', 'md.py', ['parameters.txt', 'system.chk'])


def test_example_999_back_propagation():
    check_example(__name__, '999_back_propagation', 'bp.py', [])
