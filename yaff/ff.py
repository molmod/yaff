# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
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

from yaff.ext import compute_ewald_reci, compute_ewald_corr


__all__ = [
    'ForceField', 'SumForceField', 'PairTerm', 'EwaldReciprocalTerm',
    'EwaldCorrectionTerm', 'EwaldNeutralizingTerm',
]


class ForceField(object):
    def __init__(self, system):
        self.system = system

    def update_rvecs(self, rvecs):
        self.system.update_rvecs(rvecs)

    def update_pos(self, pos):
        self.system.pos[:] = pos

    def compute(self, gradient=None):
        raise NotImplementedError


class SumForceField(ForceField):
    def __init__(self, system, terms, nlists=None):
        ForceField.__init__(self, system)
        self.terms = terms
        self.nlists = nlists
        self.needs_update = True

    def update_rvecs(self, rvecs):
        ForceField.update_rvecs(self, pos)
        self.needs_update = True

    def update_pos(self, pos):
        ForceField.update_pos(self, pos)
        self.needs_update = True

    def compute(self, gradient=None):
        if self.needs_update:
            if self.nlists is not None:
                self.nlists.update()
            self.needs_update = False
        l = [term.compute(gradient) for term in self.terms]
        print l
        return sum(l)


class PairTerm(object):
    def __init__(self, nlists, scalings, pair_pot):
        self.nlists = nlists
        self.scalings = scalings
        self.pair_pot = pair_pot
        self.nlists.request_cutoff(pair_pot.cutoff)

    def compute(self, gradient=None):
        assert len(self.nlists) == len(self.scalings)
        return sum([
            self.pair_pot.compute(i, self.nlists[i], self.scalings[i], gradient)
            for i in xrange(len(self.nlists))
        ])


class EwaldReciprocalTerm(object):
    def __init__(self, system, charges, alpha, gmax):
        assert len(system.rvecs) == 3
        self.system = system
        self.charges = charges
        self.alpha = alpha
        self.gmax = gmax

    def compute(self, gradient=None):
        return compute_ewald_reci(
            self.system.pos, self.charges, self.system.gvecs,
            self.system.volume, self.alpha, self.gmax, gradient
        )


class EwaldCorrectionTerm(object):
    def __init__(self, system, charges, alpha, scalings):
        assert len(system.rvecs) == 3
        self.system = system
        self.charges = charges
        self.alpha = alpha
        self.scalings = scalings

    def compute(self, gradient=None):
        return sum([
            compute_ewald_corr(self.system.pos, i, self.charges,
                               self.system.rvecs, self.system.gvecs, self.alpha,
                               self.scalings[i], gradient)
            for i in xrange(len(self.scalings))
        ])


class EwaldNeutralizingTerm(object):
    def __init__(self, system, charges, alpha):
        assert len(system.rvecs) == 3
        self.system = system
        self.charges = charges
        self.alpha = alpha

    def compute(self, gradient=None):
        return self.charges.sum()**2*np.pi/(2.0*system.volume*self.alpha**2)
