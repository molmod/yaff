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

from yaff import atsel_compile
from yaff.pes.ff import ForceField


__all__ = [
    'CostFunction',
    'Simulation', 'GeoOptSimulation',
    'Test', 'BondLengthTest', 'BendAngleTest',
]


class CostFunction(object):
    def __init__(self, parameter_transform, simulations):
        self.parameter_transform = parameter_transform
        self.simulations = simulations

    def __call__(self, x):
        # Modify the parameters
        parameters = self.parameter_transform(x)
        # Run simulations with the new parameters
        cost = 0.0
        for simulation in self.simulations:
            simulation(parameters)
            for test in simulation.tests:
                cost += 0.5*test.error**2
        return cost


class Simulation(object):
    def __init__(self, system, tests, **kwargs):
        self.system = system
        self.tests = tests
        self.kwargs = kwargs

    def __call__(self, parameters):
        # prepare force field
        ff = ForceField.generate(self.system, parameters, **self.kwargs)
        # run actual simulation
        output = self.run(ff)
        # run the tests on the output of the simulation
        for test in self.tests:
            test(ff, output)

    def run(self, ff):
        raise NotImplementedError


class GeoOptSimulation(Simulation):
    def run(self, ff):
        from yaff import CartesianDOF, CGOptimizer
        dof = CartesianDOF(ff)
        opt = CGOptimizer(dof)
        opt.run(1000)
        return {'pos': self.system.pos.copy()}


class Test(object):
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.error = None

    def __call__(self, ff, output):
        self.error = self.compute_error(ff, output)/self.tolerance

    def compute_error(self, ff, output):
        raise NotImplementedError


class BondLengthTest(Test):
    def __init__(self, refpos, tolerance, rule0='!0', rule1='!0'):
        if isinstance(rule0, basestring):
            rule0 = atsel_compile(rule0)
        if isinstance(rule1, basestring):
            rule1 = atsel_compile(rule1)

        self.refpos = refpos
        self.rule0 = rule0
        self.rule1 = rule1
        Test.__init__(self, tolerance)

    def compute_error(self, ff, output):
        pos = output['pos']
        sys = ff.system
        sumsq = 0.0
        count = 0
        for i0, i1 in sys.iter_bonds():
            if (self.rule0(sys, i0) and self.rule1(sys, i1)) or \
               (self.rule0(sys, i1) and self.rule1(sys, i0)):
                dist = np.linalg.norm(pos[i0] - pos[i1])
                refdist = np.linalg.norm(self.refpos[i0] - self.refpos[i1])
                from molmod import angstrom
                sumsq += (dist - refdist)**2
                count += 1
        return np.sqrt(sumsq/count)


class BendAngleTest(Test):
    def __init__(self, refpos, tolerance, rule0='!0', rule1='!0', rule2='!0'):
        if isinstance(rule0, basestring):
            rule0 = atsel_compile(rule0)
        if isinstance(rule1, basestring):
            rule1 = atsel_compile(rule1)
        if isinstance(rule2, basestring):
            rule2 = atsel_compile(rule2)

        self.refpos = refpos
        self.rule0 = rule0
        self.rule1 = rule1
        self.rule2 = rule2
        Test.__init__(self, tolerance)

    def compute_error(self, ff, output):
        def compute_angle(p0, p1, p2):
            d01 = np.linalg.norm(p0-p1)
            d21 = np.linalg.norm(p2-p1)
            cos = np.dot(p0-p1,p2-p1)/d01/d21
            cos = np.clip(cos, -1, 1)
            return np.arccos(cos)

        pos = output['pos']
        sys = ff.system
        sumsq = 0.0
        count = 0
        for i0, i1, i2 in sys.iter_angles():
            if (self.rule0(sys, i0) and self.rule1(sys, i1) and self.rule2(sys, i2)) or \
               (self.rule0(sys, i2) and self.rule1(sys, i1) and self.rule2(sys, i0)):
                angle = compute_angle(pos[i0], pos[i1], pos[i2])
                refangle = compute_angle(self.refpos[i0], self.refpos[i1], self.refpos[i2])
                sumsq += (angle - refangle)**2
                count += 1
        return np.sqrt(sumsq/count)
