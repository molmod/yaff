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
from molmod import kjmol, angstrom

from yaff import atsel_compile, log
from yaff.pes.ff import ForceField
from yaff.sampling.harmonic import estimate_cart_hessian


__all__ = [
    'CostFunction',
    'Simulation', 'GeoOptSimulation', 'GeoOptHessianSimulation',
    'ICGroup', 'BondGroup', 'BendGroup',
    'Test', 'ICTest', 'FCTest',
]


class CostFunction(object):
    def __init__(self, parameter_transform, tests):
        self.parameter_transform = parameter_transform
        self.tests = tests
        self.simulations = set([])
        for test in tests:
            self.simulations.update(test.simulations)

    def __call__(self, x):
        # Modify the parameters
        parameters = self.parameter_transform(x)
        #for counter, line in parameters['BENDCHARM']['PARS']:
        #    log('FOO %s' % line.strip())
        # Run simulations with the new parameters
        for simulation in self.simulations:
            simulation(parameters)
        cost = 0.0
        for test in self.tests:
            cost += 0.5*test()**2
        return cost


class Simulation(object):
    def __init__(self, system, **kwargs):
        self.system = system
        self.kwargs = kwargs

    def __call__(self, parameters):
        # prepare force field
        ff = ForceField.generate(self.system, parameters, **self.kwargs)
        # run actual simulation
        self.run(ff)

    def run(self, ff):
        raise NotImplementedError


class GeoOptSimulation(Simulation):
    def __init__(self, system, **kwargs):
        self.refpos = system.pos.copy()
        Simulation.__init__(self, system, **kwargs)

    def run(self, ff):
        from yaff import CartesianDOF, CGOptimizer, OptScreenLog
        #prevpos = ff.system.pos[:].copy()
        #energy0 = ff.compute()
        #ff.system.pos[:] = self.refpos
        #energy1 = ff.compute()
        #if energy1 > energy0:
        ff.system.pos[:] *= np.random.uniform(0.99, 1.01, ff.system.pos.shape)
        dof = CartesianDOF(ff, gpos_rms=1e-8)
        sl = OptScreenLog(step=10)
        opt = CGOptimizer(dof, hooks=[sl])
        opt.run(5000)


class GeoOptHessianSimulation(GeoOptSimulation):
    def run(self, ff):
        GeoOptSimulation.run(self, ff)
        self.hessian = estimate_cart_hessian(ff)


class ICGroup(object):
    natom = None

    def __init__(self, rules=None, cases=None):
        self.cases = cases
        if cases is None:
            if rules is None:
                rules = ['!0'] * self.natom
            compiled_rules = []
            for rule in rules:
                if isinstance(rule, basestring):
                    rule = atsel_compile(rule)
                compiled_rules.append(rule)
            self.rules = compiled_rules
        elif rules is not None:
            raise ValueError('Either rules are cases must be provided, not both.')

    def get_cases(self, system):
        return list(self._iter_cases(self.rules, system))

    def _iter_cases(self, rules, system):
        raise NotImplementedError

    def compute_ic(self, pos, indexes):
        raise NotImplementedError

    def compute_tangent(self, pos, indexes):
        raise NotImplementedError


class BondGroup(ICGroup):
    natom = 2

    def _iter_cases(self, rules, system):
        rule0, rule1 = rules
        for i0, i1 in system.iter_bonds():
            if (rule0(system, i0) and rule1(system, i1)) or \
               (rule0(system, i1) and rule1(system, i0)):
                yield i0, i1

    def compute_ic(self, pos, indexes):
        i0, i1 = indexes
        return np.linalg.norm(pos[i0] - pos[i1])

    def compute_tangent(self, pos, indexes):
        # TODO: implement this with the vlist code in yaff.pes
        result = np.zeros(pos.shape, float)
        p0 = pos[indexes[0]]
        p1 = pos[indexes[1]]
        delta = p0 - p1
        delta /= np.linalg.norm(delta)
        result[indexes[0]] = delta
        result[indexes[1]] = -delta
        return result.ravel()


class BendGroup(ICGroup):
    natom = 3

    def _iter_cases(self, rules, system):
        rule0, rule1, rule2 = rules
        for i0, i1, i2 in system.iter_angles():
            if (rule0(system, i0) and rule1(system, i1) and rule2(system, i2)) or \
               (rule0(system, i2) and rule1(system, i1) and rule2(system, i0)):
                yield i0, i1, i2

    def compute_ic(self, pos, indexes):
        i0, i1, i2 = indexes
        d01 = pos[i0]-pos[i1]
        d21 = pos[i2]-pos[i1]
        cos = np.dot(d01,d21)/np.linalg.norm(d01)/np.linalg.norm(d21)
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos)


class Test(object):
    def __init__(self, tolerance, simulations):
        self.tolerance = tolerance
        self.simulations = simulations

    def __call__(self):
        # Compute a dimensionless error
        return self.compute_error()/self.tolerance

    def compute_error(self):
        raise NotImplementedError


class ICTest(Test):
    def __init__(self, tolerance, refpos, simulation, icgroup):
        # assign attributes
        self.refpos = refpos
        self.simulation = simulation
        self.icgroup = icgroup
        self.cases = icgroup.get_cases(simulation.system)
        # precompute internal coordinates of reference pos
        refics = []
        for indexes in self.cases:
            refics.append(icgroup.compute_ic(refpos, indexes))
        self.refics = np.array(refics)
        # Call super class
        Test.__init__(self, tolerance, [simulation])

    def compute_error(self):
        sumsq = 0.0
        count = 0
        pos = self.simulation.system.pos
        for i in xrange(len(self.cases)):
            indexes = self.cases[i]
            sumsq += (self.refics[i] - self.icgroup.compute_ic(pos, indexes))**2
            count += 1
        return np.sqrt(sumsq/count)


class FCTest(Test):
    """A test for force constants

       This is special in the sense that the force constants derived from the
       hessian are not sensitive to either orientation of the molecule or the
       choice of other internal coordinates. This is realized by comparing the
       second order derivatives along PES scanes of selected internal
       coordinates.
    """
    def __init__(self, tolerance, refpos, refhessian, simulation, icgroup):
        # assign attributes
        self.refpos = refpos
        self.refhessian = refhessian
        self.simulation = simulation
        self.icgroup = icgroup
        self.cases = icgroup.get_cases(simulation.system)
        # precompute internal coordinates of reference pos
        reffcs = []
        for indexes in self.cases:
            reffcs.append(self.compute_fc(refpos, refhessian, indexes))
        self.reffcs = np.array(reffcs)
        print self.reffcs/(kjmol/angstrom**2)
        # Call super class
        Test.__init__(self, tolerance, [simulation])

    def compute_error(self):
        sumsq = 0.0
        count = 0
        pos = self.simulation.system.pos
        hessian = self.simulation.hessian
        for i in xrange(len(self.cases)):
            indexes = self.cases[i]
            print self.compute_fc(pos, hessian, indexes)/(kjmol/angstrom**2)
            sumsq += (self.reffcs[i] - self.compute_fc(pos, hessian, indexes))**2
            count += 1
        return np.sqrt(sumsq/count)

    def compute_fc(self, pos, hessian, indexes):
        # the derivative of the internal coordinate toward Cartesian coordinates
        tangent = self.icgroup.compute_tangent(pos, indexes)
        # take a pseudo-inverse of the hessian through the eigen decomposition
        evals, evecs = np.linalg.eigh(hessian)
        # prune near-zero and negative eigenvalues from the Hessian of the
        # environment
        mask = evals > 1*kjmol/angstrom**2
        evals = evals[mask]
        evecs = evecs[:,mask]
        # compute force constant
        tmp = np.dot(tangent, evecs)
        return 1.0/(tmp**2/evals).sum()
