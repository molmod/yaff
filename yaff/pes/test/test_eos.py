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

import numpy as np
import pkg_resources

from molmod.units import kelvin, bar

from yaff import *

def test_idealgas():
    ig = IdealGas()
    P = 0.15
    fugacity = ig.calculate_fugacity(205,P)
    assert fugacity==P


def test_vdw_methane():
    Tc = 190.6*kelvin
    Pc = 46.04*bar
    a = 27.0*(boltzmann*Tc)**2/64.0/Pc
    b = boltzmann*Tc/8.0/Pc
    eos = vdWEOS(a, b)
    # Following data come from http://people.ds.cam.ac.uk/pjb10/thermo/pure.html
    reference = [
        (250*kelvin,5*bar,0.9838),
        (125*kelvin,8*bar,0.8622),
        (185*kelvin,40.0*bar,0.7174),
    ]
    for T, P, phi in reference:
        f = eos.calculate_fugacity(T, P)
#        print("Fugacity: ref = %8.5f bar computed = %8.5f bar" % (phi*P/bar, f/bar))
        assert np.abs(phi*P-f)<1e-2*bar


def test_preos_co2():
    Tc, Pc, omega = 304.2*kelvin, 73.82*bar, 0.228
    eos = PREOS(Tc, Pc, omega)
    # Following data come from http://people.ds.cam.ac.uk/pjb10/thermo/pure.html
    reference = [
        (200*kelvin,1.5*bar,0.9748),
        (400*kelvin,0.8*bar,0.9983),
        (305*kelvin,40.0*bar,0.8058),
    ]
    for T, P, phi in reference:
        f = eos.calculate_fugacity(T, P)
#        print("Fugacity: ref = %8.5f bar computed = %8.5f bar" % (phi*P/bar, f/bar))
        assert np.abs(phi*P-f)<1e-2*bar


def test_preos_from_name():
    eos = PREOS.from_name('carbondioxide')
    assert eos.Tc == 304.21*kelvin
    assert eos.Pc == 7.383*1e6*pascal
    assert eos.omega == 0.2236
    assert eos.mass == 44.010*amu