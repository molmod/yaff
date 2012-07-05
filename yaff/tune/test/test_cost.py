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
from molmod.io import load_chk


def test_water_cost_dist_ic():
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

    simulations = [GeoOptSimulation(system)]
    tests = [ICTest(0.1*angstrom, refpos, simulations[0], BondGroup())]
    cost = CostFunction(pt, tests)

    x = np.array([1.0])
    assert abs(cost(x) - 0.5*((1.0238240000e+00 - 1.1)/0.1)**2) < 1e-5

    x = np.array([1.1])
    assert abs(cost(x) - 0.5*((1.1*1.0238240000e+00 - 1.1)/0.1)**2) < 1e-5

    x = np.array([0.8])
    assert abs(cost(x) - 0.5*((0.8*1.0238240000e+00 - 1.1)/0.1)**2) < 1e-5


def test_water_cost_dist_fc():
    sample = load_chk('input/water_hessian.chk')
    system = System(pos=sample['pos'], numbers=sample['numbers'], ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    parameters = Parameters.from_file('input/parameters_water.txt')
    del parameters.sections['FIXQ']
    del parameters.sections['DAMPDISP']
    del parameters.sections['EXPREP']

    rules = [ScaleRule('BONDFUES', 'PARS', 'O\s*H', 2)]
    mods = [ParameterModifier(rules)]
    pt = ParameterTransform(parameters, mods)

    simulations = [GeoOptHessianSimulation(system)]
    tests = [FCTest(kjmol/angstrom**2, sample['pos'], sample['hessian'].reshape(9, 9), simulations[0], BondGroup())]
    cost = CostFunction(pt, tests)

    x = np.array([1.0])
    assert abs(cost(x) - 0.5*(5159.1871966 - 4.0088096730e+03)**2) < 1

    x = np.array([1.1])
    assert abs(cost(x) - 0.5*(5159.1871966 - 1.1*4.0088096730e+03)**2) < 1

    x = np.array([0.8])
    assert abs(cost(x) - 0.5*(5159.1871966 - 0.8*4.0088096730e+03)**2) < 1


def test_water_cost_angle_ic():
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

    simulations = [GeoOptSimulation(system)]
    tests = [ICTest(5*deg, refpos, simulations[0], BendGroup())]
    cost = CostFunction(pt, tests)

    x = np.array([1.0])
    assert abs(cost(x) - 0.5*((np.arccos(2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2) < 1e-4

    x = np.array([1.1])
    assert abs(cost(x) - 0.5*((np.arccos(1.1*2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2) < 1e-4

    x = np.array([0.8])
    assert abs(cost(x) - 0.5*((np.arccos(0.8*2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2) < 1e-4
