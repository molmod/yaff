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


__all__ = ['ForceField', 'SumForceField', 'PairTerm']


class ForceField(object):
    def __init__(self, system):
        self.system = system

    def update_pos(self, pos):
        system.pos[:] = pos

    def energy(self):
        raise NotImplementedError

    def energy_gradient(self):
        raise NotImplementedError


class SumForceField(ForceField):
    def __init__(self, system, terms, nlists=None):
        ForceField.__init__(self, system)
        self.terms = terms
        self.nlists = nlists

    def update_pos(self, pos):
        ForceField.update_pos(self, pos)
        if self.nlists is not None:
            self.nlists.update()

    def energy(self):
        result = 0
        for term in self.terms:
            result += term.energy()

    def energy_gradient(self):
        energy = 0
        gradient = 0
        for term in self.terms:
            e, g = term.energy()
            energy += e
            gradient += g
        return energy, gradient


class PairTerm(object):
    def __init__(self, nlists, scalings, pairpot):
        self.nlists = nlists
        self.scalings = scalings
        self.pairpot = pairpot
        self.nlists.request_cutoff(pairpot.get_cutoff())

    def energy(self):
        result = 0
        for i in 0,:#xrange(len(self.nlists)):
            result += self.pairpot.energy(i, self.nlists[i], self.scalings[i])
        return result
