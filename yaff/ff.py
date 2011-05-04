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
from yaff.dlist import DeltaList
from yaff.iclist import InternalCoordinateList
from yaff.vlist import ValenceList


__all__ = [
    'ForcePart', 'ForceField', 'PairPart', 'EwaldReciprocalPart',
    'EwaldCorrectionPart', 'EwaldNeutralizingPart', 'ValencePart',
]


class ForcePart(object):
    def __init__(self, system):
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()

    def clear(self):
        # Fill in bogus values that make things crash and burn
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        self.clear()

    def update_pos(self, pos):
        self.clear()

    def compute(self, gpos=None, vtens=None):
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = self.gpos
            my_gpos[:] = 0.0
        if vtens is None:
            my_vtens = None
        else:
            my_vtens = self.vtens
            my_vtens[:] = 0.0
        self.energy = self._internal_compute(my_gpos, my_vtens)
        if gpos is not None:
            gpos += my_gpos
        if vtens is not None:
            vtens += my_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        raise NotImplementedError


class ForceField(ForcePart):
    def __init__(self, system, parts, nlists=None):
        ForcePart.__init__(self, system)
        self.system = system
        self.parts = parts
        self.nlists = nlists
        self.needs_nlists_update = True

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)
        self.needs_nlists_update = True

    def update_pos(self, pos):
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        self.needs_nlists_update = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlists_update:
            self.nlists.update()
            self.needs_nlists_update = False
        return sum([part.compute(gpos, vtens) for part in self.parts])


class PairPart(ForcePart):
    def __init__(self, system, nlists, scalings, pair_pot):
        ForcePart.__init__(self, system)
        self.nlists = nlists
        self.scalings = scalings
        self.pair_pot = pair_pot
        self.nlists.request_rcut(pair_pot.rcut)

    def _internal_compute(self, gpos, vtens):
        assert len(self.nlists) == len(self.scalings)
        result = 0.0
        for i in xrange(len(self.nlists)):
            result += self.pair_pot.compute(i, self.nlists[i], self.scalings[i], gpos, vtens)
        return result


class EwaldReciprocalPart(ForcePart):
    def __init__(self, system, charges, alpha, gcut=0.35):
        ForcePart.__init__(self, system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        self.system = system
        self.charges = charges
        self.alpha = alpha
        self.gcut = gcut
        self.work = np.empty(system.natom*2)
        self.needs_update_gmax = True

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.needs_update_gmax = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_update_gmax:
            self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
            self.needs_update_gmax = False
        energy = compute_ewald_reci(
            self.system.pos, self.charges, self.system.cell, self.alpha,
            self.gmax, self.gcut, gpos, self.work, vtens
        )
        return energy


class EwaldCorrectionPart(ForcePart):
    def __init__(self, system, charges, alpha, scalings):
        ForcePart.__init__(self, system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        self.system = system
        self.charges = charges
        self.alpha = alpha
        self.scalings = scalings

    def _internal_compute(self, gpos, vtens):
        return sum([
            compute_ewald_corr(self.system.pos, i, self.charges,
                               self.system.cell, self.alpha, self.scalings[i],
                               gpos, vtens)
            for i in xrange(len(self.scalings))
        ])


class EwaldNeutralizingPart(ForcePart):
    def __init__(self, system, charges, alpha):
        ForcePart.__init__(self, system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        self.system = system
        self.charges = charges
        self.alpha = alpha

    def _internal_compute(self, gpos, vtens):
        fac = self.charges.sum()**2*np.pi/(2.0*self.system.cell.volume*self.alpha**2)
        if vtens is not None:
            vtens.ravel()[::4] -= fac
        return fac


class ValencePart(ForcePart):
    def __init__(self, system):
        ForcePart.__init__(self, system)
        self.dlist = DeltaList(system)
        self.iclist = InternalCoordinateList(self.dlist)
        self.vlist = ValenceList(self.iclist)

    def add_term(self, term):
        self.vlist.add_term(term)

    def _internal_compute(self, gpos, vtens):
        self.dlist.forward()
        self.iclist.forward()
        energy = self.vlist.forward()
        if not ((gpos is None) and (vtens is None)):
            self.vlist.back()
            self.iclist.back()
            self.dlist.back(gpos, vtens)
        return energy
