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
'''YAFF - Yet another force field

   The ``yaff`` package contains the subpackages that define the main
   functionalities in yaff: force field models (:mod:`yaff.pes`), sampling
   (:mod:`yaff.sampling`), trajectory analysis (:mod:`yaff.analysis`) and
   parameter tuning (:mod:`yaff.tune`). These major subpackages are discusses in
   the following sections.
'''


from .version import __version__

from molmod.units import *
from molmod.constants import *

from yaff.analysis import *
from yaff.atselect import *
from yaff.conversion import *
from yaff.log import *
from yaff.pes import *
from yaff.sampling import *
from yaff.system import *
from yaff.tune import *
