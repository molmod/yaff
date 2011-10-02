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

from yaff.log import log
from yaff.pes.ext import compute_ewald_reci, compute_ewald_corr, PairPotEI, \
    PairPotLJ, PairPotMM3, PairPotGrimme
from yaff.pes.dlist import DeltaList
from yaff.pes.iclist import InternalCoordinateList
from yaff.pes.vlist import ValenceList


__all__ = [
    'ForcePart', 'ForceField', 'ForcePartPair', 'ForcePartEwaldReciprocal',
    'ForcePartEwaldCorrection', 'ForcePartEwaldNeutralizing', 'ForcePartValence',
]


class ForcePart(object):
    def __init__(self, name, system):
        """
           **Arguments:**

           name
                A name for this part of the force field. This name must adhere
                to the following conventions: all lower case, no white space,
                and short. It is used to construct part_* attributes in the
                ForceField class, where * is the name.

           system
                The system where this part of the FF is applied to.
        """
        self.name = name
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()

    def clear(self):
        """Fill in bogus values that make things crash and burn"""
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):

        self.clear()

    def update_pos(self, pos):
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """Compute the energy and optionally some derivatives for this FF (part)

           **Optional arguments:**

           gpos
                The derivatives of the energy towards the Cartesian coordinates
                of the atoms. ('g' stands for gradient and 'pos' for positions.)
                This must be a writeable numpy array with shape (N, 3) where N
                is the number of atoms.

           vtens
                The force contribution to the pressure tensor. This is also
                known as the virial tensor. It represents the derivative of the
                energy towards uniform deformations, including changes in the
                shape of the unit cell. (v stands for virial and 'tens' stands
                for tensor.) This must be a writeable numpy array with shape (3,
                3).

           The energy is returned. The optional arguments are Fortran-style
           output arguments. When they are present, the corresponding results
           are computed and **added** to the current contents of the array.
        """
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
        """
           **Arguments:**

           system
                An instance of the System class.

           parts
                A list of instances of sublcasses of ForcePart. These are
                the different types of contributions to the force field, e.g.
                valence interactions, real-space electrostatics, and so on.

           **Optional arguments:**

           nlists
                A NeighborLists instance. This is only required if some parts
                use this.
        """
        ForcePart.__init__(self, 'all', system)
        self.system = system
        self.parts = parts
        self.nlists = nlists
        self.needs_nlists_update = nlists is not None
        # Make the parts also accessible as simple attributes.
        for part in parts:
            name = 'part_%s' % part.name
            if name in self.__dict__:
                raise ValueError('The part %s occurs twice in the force field.' % name)
            self.__dict__[name] = part

    @classmethod
    def generate(cls, system, fn_parameters, **kwargs):
        """Create a force field for the given system with the given parameters.

           **Arguments:**

           system
                An instance of the System class

           fn_parameters
                The filename of the parameter file. This must be a text file
                that adheres to YAFF parameter format.

           See the constructor of the FFArgs class for the available optional
           arguments.

           This method takes care of setting up the FF object, and configuring
           all the necessary FF parts. This is a lot easier than creating an FF
           with the default constructor. Parameters for atom types that are not
           present in the system, are simply ignored.
        """
        from yaff.pes.generator import ParsedPars, generators, FFArgs
        log.enter('GEN')
        parsed_pars = ParsedPars(fn_parameters)
        ff_args = FFArgs(**kwargs)
        for prefix in parsed_pars.info:
            generator = generators.get(prefix)
            if generator is None:
                if log.do_warning:
                    log.warn('There is no generator named %s.' % prefix)
            else:
                generator(system, parsed_pars.get_section(prefix), ff_args)
        log.leave()
        return ForceField(system, ff_args.parts, ff_args.nlists)

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)
        if self.nlists is not None:
            self.needs_nlists_update = True

    def update_pos(self, pos):
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        if self.nlists is not None:
            self.needs_nlists_update = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlists_update:
            self.nlists.update()
            self.needs_nlists_update = False
        return sum([part.compute(gpos, vtens) for part in self.parts])


class ForcePartPair(ForcePart):
    def __init__(self, system, nlists, scalings, pair_pot):
        ForcePart.__init__(self, 'pair_%s' % pair_pot.name, system)
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


class ForcePartEwaldReciprocal(ForcePart):
    def __init__(self, system, alpha, gcut=0.35):
        ForcePart.__init__(self, 'ewald_reci', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        self.needs_update_gmax = True

    def update_gmax(self):
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        energy = compute_ewald_reci(
            self.system.pos, self.system.charges, self.system.cell, self.alpha,
            self.gmax, self.gcut, gpos, self.work, vtens
        )
        return energy


class ForcePartEwaldCorrection(ForcePart):
    def __init__(self, system, alpha, scalings):
        ForcePart.__init__(self, 'ewald_cor', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.scalings = scalings

    def _internal_compute(self, gpos, vtens):
        return sum([
            compute_ewald_corr(self.system.pos, i, self.system.charges,
                               self.system.cell, self.alpha, self.scalings[i],
                               gpos, vtens)
            for i in xrange(len(self.scalings))
        ])


class ForcePartEwaldNeutralizing(ForcePart):
    def __init__(self, system, alpha):
        ForcePart.__init__(self, 'ewald_neut', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha

    def _internal_compute(self, gpos, vtens):
        fac = self.system.charges.sum()**2*np.pi/(2.0*self.system.cell.volume*self.alpha**2)
        if vtens is not None:
            vtens.ravel()[::4] -= fac
        return fac


class ForcePartValence(ForcePart):
    def __init__(self, system):
        ForcePart.__init__(self, 'valence', system)
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
