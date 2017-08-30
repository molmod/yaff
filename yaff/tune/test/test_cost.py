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


import numpy as np
import pkg_resources

from yaff import *
from yaff.test.common import get_system_peroxide
from molmod.io import load_chk


def test_icgroup_cases():
    sys = get_system_peroxide()
    assert (BondGroup(sys).cases == sys.bonds).all()
    assert BondGroup(sys, cases=[[1,2], [1,3]]).cases == [[1,2],[1,3]]
    assert BondGroup(sys, rules=['8', '8']).cases == [[0,1]]
    assert BondGroup(sys, rules=['1', '8']).cases == [[0,2],[1,3]]
    assert BendGroup(sys).cases == [[2, 0, 1], [3, 1, 0]]
    assert BendGroup(sys, cases=[[1,2,0], [0,1,3]]).cases == [[1,2,0],[0,1,3]]


def test_water_cost_dist_ic():
    fn_xyz = pkg_resources.resource_filename(__name__, '../../data/test/water_trajectory.xyz')
    system = System.from_file(fn_xyz, ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    parameters = Parameters.from_file(fn_pars)
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

    simulations = [GeoOptSimulation('only', system)]
    tests = [ICTest(0.1*angstrom, refpos, simulations[0], BondGroup(system))]
    assert tests[0].icgroup.cases == [[1, 0], [2, 0]]
    cost = CostFunction(pt, {'all': tests})

    x = np.array([1.0])
    assert abs(cost(x) - np.log(0.5*((1.0238240000e+00 - 1.1)/0.1)**2)) < 1e-5

    x = np.array([1.1])
    assert abs(cost(x) - np.log(0.5*((1.1*1.0238240000e+00 - 1.1)/0.1)**2)) < 1e-5

    x = np.array([0.8])
    assert abs(cost(x) - np.log(0.5*((0.8*1.0238240000e+00 - 1.1)/0.1)**2)) < 1e-5

    results = {}
    parameters = cost.parameter_transform(x)
    for simulation in simulations:
        results[simulation.name] = simulation(parameters)
    my_results = tests[0].filter_results(results)
    assert results == my_results


def test_water_cost_dist_fc():
    fn_chk = pkg_resources.resource_filename(__name__, '../../data/test/water_hessian.chk')
    sample = load_chk(fn_chk)
    system = System(pos=sample['pos'], numbers=sample['numbers'], ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    parameters = Parameters.from_file(fn_pars)
    del parameters.sections['FIXQ']
    del parameters.sections['DAMPDISP']
    del parameters.sections['EXPREP']

    rules = [ScaleRule('BONDFUES', 'PARS', 'O\s*H', 2)]
    mods = [ParameterModifier(rules)]
    pt = ParameterTransform(parameters, mods)

    simulations = [GeoOptHessianSimulation('only', system)]
    tests = [FCTest(kjmol/angstrom**2, sample['pos'], sample['hessian'].reshape(9, 9), simulations[0], BondGroup(system))]
    assert tests[0].icgroup.cases == [[1, 0], [2, 0]]
    cost = CostFunction(pt, {'all': tests})

    x = np.array([1.0])
    assert abs(cost(x) - np.log(0.5*(5159.1871966 - 4.0088096730e+03)**2)) < 1

    x = np.array([1.1])
    assert abs(cost(x) - np.log(0.5*(5159.1871966 - 1.1*4.0088096730e+03)**2)) < 1

    x = np.array([0.8])
    assert abs(cost(x) - np.log(0.5*(5159.1871966 - 0.8*4.0088096730e+03)**2)) < 1


def test_water_cost_angle_ic():
    fn_xyz = pkg_resources.resource_filename(__name__, '../../data/test/water_trajectory.xyz')
    system = System.from_file(fn_xyz, ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    parameters = Parameters.from_file(fn_pars)
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

    simulations = [GeoOptSimulation('only', system)]
    tests = [ICTest(5*deg, refpos, simulations[0], BendGroup(system))]
    assert tests[0].icgroup.cases == [[2, 0, 1]]
    cost = CostFunction(pt, {'all': tests})

    x = np.array([1.0])
    assert abs(cost(x) - np.log(0.5*((np.arccos(2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2)) < 1e-4

    x = np.array([1.1])
    assert abs(cost(x) - np.log(0.5*((np.arccos(1.1*2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2)) < 1e-4

    x = np.array([0.8])
    assert abs(cost(x) - np.log(0.5*((np.arccos(0.8*2.7892000007e-02) - 1.5707963267948966)/(5*deg))**2)) < 1e-4


def test_water_cost_angle_fc():
    fn_chk = pkg_resources.resource_filename(__name__, '../../data/test/water_hessian.chk')
    sample = load_chk(fn_chk)
    system = System(pos=sample['pos'], numbers=sample['numbers'], ffatypes=['O', 'H', 'H'])
    system.detect_bonds()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    parameters = Parameters.from_file(fn_pars)
    del parameters.sections['FIXQ']
    del parameters.sections['DAMPDISP']
    del parameters.sections['EXPREP']

    rules = [ScaleRule('BENDCHARM', 'PARS', 'O\s*H', 3)]
    mods = [ParameterModifier(rules)]
    pt = ParameterTransform(parameters, mods)

    simulations = [GeoOptHessianSimulation('only', system)]
    tests = [FCTest(kjmol, sample['pos'], sample['hessian'].reshape(9, 9), simulations[0], BendGroup(system))]
    assert tests[0].icgroup.cases == [[2, 0, 1]]
    cost = CostFunction(pt, {'all': tests})

    x = np.array([1.0])
    assert abs(cost(x) - np.log(0.5*(394.59354836 - 302.068346061)**2)) < 1

    x = np.array([1.1])
    assert abs(cost(x) - np.log(0.5*(394.59354836 - 1.1*302.068346061)**2)) < 1

    x = np.array([0.8])
    assert abs(cost(x) - np.log(0.5*(394.59354836 - 0.8*302.068346061)**2)) < 1
