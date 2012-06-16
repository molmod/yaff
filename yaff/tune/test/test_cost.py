# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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

from yaff import *


def test_water_cost_dist():
    system = System.from_file('input/water_trajectory.xyz', ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    parameters = Parameters.from_file('input/parameters_water.txt')
    del parameters.sections['FIXQ']
    del parameters.sections['DAMPDISP']
    del parameters.sections['EXPREP']

    refpos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.1],
        [0.0, 1.1, 0.0],
    ])*angstrom

    rules = [ScaleRule('BONDFUES', 'PARS', 'O\s*H', 3)]
    mods = [ParameterModifier(rules)]
    pt = ParameterTransform(parameters, mods)

    tests = [BondLengthTest(refpos, 0.1*angstrom)]
    simulations = [GeoOptSimulation(system, tests)]
    cost = CostFunction(pt, simulations)

    x = np.array([1.0])
    assert abs(cost(x) - 0.5*((1.0238240000e+00 - 1.1)/0.1)**2) < 1e-5

    x = np.array([1.1])
    assert abs(cost(x) - 0.5*((1.1*1.0238240000e+00 - 1.1)/0.1)**2) < 1e-5

    x = np.array([0.8])
    assert abs(cost(x) - 0.5*((0.8*1.0238240000e+00 - 1.1)/0.1)**2) < 1e-5


def test_water_cost_angle():
    system = System.from_file('input/water_trajectory.xyz', ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    parameters = Parameters.from_file('input/parameters_water.txt')
    del parameters.sections['FIXQ']
    del parameters.sections['DAMPDISP']
    del parameters.sections['EXPREP']

    refpos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.1],
        [0.0, 1.1, 0.0],
    ])*angstrom

    rules = [ScaleRule('BENDCHARM', 'PARS', 'H\s*O\s*H', 4)]
    mods = [ParameterModifier(rules)]
    pt = ParameterTransform(parameters, mods)

    tests = [BendAngleTest(refpos, 5*deg)]
    simulations = [GeoOptSimulation(system, tests)]
    cost = CostFunction(pt, simulations)

    x = np.array([1.0])
    assert abs(cost(x) - 0.5*((np.arccos(2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2) < 1e-4

    x = np.array([1.1])
    assert abs(cost(x) - 0.5*((np.arccos(1.1*2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2) < 1e-4

    x = np.array([0.8])
    assert abs(cost(x) - 0.5*((np.arccos(0.8*2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2) < 1e-4
