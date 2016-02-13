# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
'''Force field models

   This module contains the force field computation interface that is used by
   the :mod:`yaff.sampling` package.

   The ``ForceField`` class is the main item in this module. It acts as
   container for instances of subclasses of ``ForcePart``. Each ``ForcePart``
   subclass implements a typical contribution to the force field energy, e.g.
   ``ForcePartValence`` computes covalent interactions, ``ForcePartPair``
   computes pairwise (non-bonding) interactions, and so on. The ``ForceField``
   object also contains one neighborlist object, which is used by all
   ``ForcePartPair`` objects. Actual computations are done through the
   ``compute`` method of the ``ForceField`` object, which calls the ``compute``
   method of all the ``ForceParts`` and adds up the results.
'''


import numpy as np


from yaff.log import log, timer
from yaff.pes.ext import compute_ewald_reci, compute_ewald_reci_dd, compute_ewald_corr, \
    compute_ewald_corr_dd, PairPotEI, PairPotLJ, PairPotMM3, PairPotGrimme, compute_grid3d
from yaff.pes.dlist import DeltaList
from yaff.pes.iclist import InternalCoordinateList
from yaff.pes.vlist import ValenceList
__all__ = [
    'ForcePart', 'ForceField', 'ForcePartPair', 'ForcePartEwaldReciprocal',
    'ForcePartEwaldReciprocalDD', 'ForcePartEwaldCorrectionDD',
    'ForcePartEwaldCorrection', 'ForcePartEwaldNeutralizing',
    'ForcePartValence', 'ForcePartPressure', 'ForcePartGrid',
]


class ForcePart(object):
    '''Base class for anything that can compute energies (and optionally gradient
       and virial) for a ``System`` object.
    '''
    def __init__(self, name, system):
        """
           **Arguments:**

           name
                A name for this part of the force field. This name must adhere
                to the following conventions: all lower case, no white space,
                and short. It is used to construct part_* attributes in the
                ForceField class, where * is the name.

           system
                The system to which this part of the FF applies.
        """
        self.name = name
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()

    def clear(self):
        """Fill in nan values in the cached results to indicate that they have
           become invalid.
        """
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        '''Let the ``ForcePart`` object know that the cell vectors have changed.

           **Arguments:**

           rvecs
                The new cell vectors.
        '''
        self.clear()

    def update_pos(self, pos):
        '''Let the ``ForcePart`` object know that the atomic positions have changed.

           **Arguments:**

           pos
                The new atomic coordinates.
        '''
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """Compute the energy and optionally some derivatives for this FF (part)

           The only variable inputs for the compute routine are the atomic
           positions and the cell vectors, which can be changed through the
           ``update_rvecs`` and ``update_pos`` methods. All other aspects of
           a force field are considered to be fixed between subsequent compute
           calls. If changes other than positions or cell vectors are needed,
           one must construct new ``ForceField`` and/or ``ForcePart`` objects.

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
        if np.isnan(self.energy):
            raise ValueError('The energy is not-a-number (nan).')
        if gpos is not None:
            if np.isnan(my_gpos).any():
                raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
            gpos += my_gpos
        if vtens is not None:
            if np.isnan(my_vtens).any():
                raise ValueError('Some vtens element(s) is/are not-a-number (nan).')
            vtens += my_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        '''Subclasses implement their compute code here.'''
        raise NotImplementedError


class ForceField(ForcePart):
    '''A complete force field model.'''
    def __init__(self, system, parts, nlist=None):
        """
           **Arguments:**

           system
                An instance of the ``System`` class.

           parts
                A list of instances of sublcasses of ``ForcePart``. These are
                the different types of contributions to the force field, e.g.
                valence interactions, real-space electrostatics, and so on.

           **Optional arguments:**

           nlist
                A ``NeighborList`` instance. This is required if some items in the
                parts list use this nlist object.
        """
        ForcePart.__init__(self, 'all', system)
        self.system = system
        self.parts = []
        self.nlist = nlist
        self.needs_nlist_update = nlist is not None
        for part in parts:
            self.add_part(part)
        if log.do_medium:
            with log.section('FFINIT'):
                log('Force field with %i parts:&%s.' % (
                    len(self.parts), ', '.join(part.name for part in self.parts)
                ))
                log('Neighborlist present: %s' % (self.nlist is not None))

    def add_part(self, part):
        self.parts.append(part)
        # Make the parts also accessible as simple attributes.
        name = 'part_%s' % part.name
        if name in self.__dict__:
            raise ValueError('The part %s occurs twice in the force field.' % name)
        self.__dict__[name] = part

    @classmethod
    def generate(cls, system, parameters, **kwargs):
        """Create a force field for the given system with the given parameters.

           **Arguments:**

           system
                An instance of the System class

           parameters
                Three types are accepted: (i) the filename of the parameter
                file, which is a text file that adheres to YAFF parameter
                format, (ii) a list of such filenames, or (iii) an instance of
                the Parameters class.

           See the constructor of the :class:`yaff.pes.generator.FFArgs` class
           for the available optional arguments.

           This method takes care of setting up the FF object, and configuring
           all the necessary FF parts. This is a lot easier than creating an FF
           with the default constructor. Parameters for atom types that are not
           present in the system, are simply ignored.
        """
        if system.ffatype_ids is None:
            raise ValueError('The generators needs ffatype_ids in the system object.')
        with log.section('GEN'), timer.section('Generator'):
            from yaff.pes.generator import apply_generators, FFArgs
            from yaff.pes.parameters import Parameters
            if log.do_medium:
                log('Generating force field from %s' % str(parameters))
            if not isinstance(parameters, Parameters):
                parameters = Parameters.from_file(parameters)
            ff_args = FFArgs(**kwargs)
            apply_generators(system, parameters, ff_args)
            return ForceField(system, ff_args.parts, ff_args.nlist)

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)
        if self.nlist is not None:
            self.nlist.update_rmax()
            self.needs_nlist_update = True

    def update_pos(self, pos):
        '''See :meth:`yaff.pes.ff.ForcePart.update_pos`'''
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        if self.nlist is not None:
            self.needs_nlist_update = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlist_update:
            self.nlist.update()
            self.needs_nlist_update = False
        result = sum([part.compute(gpos, vtens) for part in self.parts])
        return result


class ForcePartPair(ForcePart):
    '''A pairwise (short-range) non-bonding interaction term.

       This part can be used for the short-range electrostatics, Van der Waals
       terms, etc. Currently, one has to use multiple ``ForcePartPair``
       objects in a ``ForceField`` in order to combine different types of pairwise
       energy terms, e.g. to combine an electrostatic term with a Van der
       Waals term. (This may be changed in future to improve the computational
       efficiency.)
    '''
    def __init__(self, system, nlist, scalings, pair_pot):
        '''
           **Arguments:**

           system
                The system to which this pairwise interaction applies.

           nlist
                A ``NeighborList`` object. This has to be the same as the one
                passed to the ForceField object that contains this part.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.

           pair_pot
                An instance of the ``PairPot`` built-in class from
                :mod:`yaff.pes.ext`.
        '''
        ForcePart.__init__(self, 'pair_%s' % pair_pot.name, system)
        self.nlist = nlist
        self.scalings = scalings
        self.pair_pot = pair_pot
        self.nlist.request_rcut(pair_pot.rcut)
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()
                log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                log('  real space cutoff: %s' % log.length(pair_pot.rcut))
                tr = pair_pot.get_truncation()
                if tr is None:
                    log('  truncation:     none')
                else:
                    log('  truncation:     %s' % tr.get_log())
                self.pair_pot.log()
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section('PP %s' % self.pair_pot.name):
            return self.pair_pot.compute(self.nlist.neighs, self.scalings.stab, gpos, vtens, self.nlist.nneigh)


class ForcePartEwaldReciprocal(ForcePart):
    '''The long-range contribution to the electrostatic interaction in 3D
       periodic systems.
    '''
    def __init__(self, system, alpha, gcut=0.35, dielectric=1.0):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           **Optional arguments:**

           gcut
                The cutoff in reciprocal space.

           dielectric
                The scalar relative permittivity of the system.
        '''
        ForcePart.__init__(self, 'ewald_reci', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.dielectric = dielectric
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()
                log('  alpha:                 %s' % log.invlength(self.alpha))
                log('  gcut:                  %s' % log.invlength(self.gcut))
                log('  relative permittivity: %5.3f' % self.dielectric)
                log.hline()


    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
        if log.do_debug:
            with log.section('EWALD'):
                log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Ewald reci.'):
            return compute_ewald_reci(
                self.system.pos, self.system.charges, self.system.cell, self.alpha,
                self.gmax, self.gcut, self.dielectric, gpos, self.work, vtens
            )


class ForcePartEwaldReciprocalDD(ForcePart):
    '''The long-range contribution to the dipole-dipole
       electrostatic interaction in 3D periodic systems.
    '''
    def __init__(self, system, alpha, gcut=0.35):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           gcut
                The cutoff in reciprocal space.
        '''
        ForcePart.__init__(self, 'ewald_reci', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        if system.dipoles is None:
            raise ValueError('The system does not have dipoles.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()
                log('  alpha:             %s' % log.invlength(self.alpha))
                log('  gcut:              %s' % log.invlength(self.gcut))
                log.hline()


    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
        if log.do_debug:
            with log.section('EWALD'):
                log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Ewald reci.'):
            return compute_ewald_reci_dd(
                self.system.pos, self.system.charges, self.system.dipoles, self.system.cell, self.alpha,
                self.gmax, self.gcut, gpos, self.work, vtens
            )


class ForcePartEwaldCorrection(ForcePart):
    '''Correction for the double counting in the long-range term of the Ewald sum.

       This correction is only needed if scaling rules apply to the short-range
       electrostatics.
    '''
    def __init__(self, system, alpha, scalings, dielectric=1.0):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.

           **Optional arguments:**

           dielectric
                The scalar relative permittivity of the system.
        '''
        ForcePart.__init__(self, 'ewald_cor', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.dielectric = dielectric
        self.scalings = scalings
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()
                log('  alpha:             %s' % log.invlength(self.alpha))
                log('  relative permittivity   %5.3f' % self.dielectric)
                log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Ewald corr.'):
            return compute_ewald_corr(
                self.system.pos, self.system.charges, self.system.cell,
                self.alpha, self.scalings.stab, self.dielectric, gpos, vtens
            )


class ForcePartEwaldCorrectionDD(ForcePart):
    '''Correction for the double counting in the long-range term of the Ewald sum.

       This correction is only needed if scaling rules apply to the short-range
       electrostatics.
    '''
    def __init__(self, system, alpha, scalings):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.
        '''
        ForcePart.__init__(self, 'ewald_cor', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.scalings = scalings
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()
                log('  alpha:             %s' % log.invlength(self.alpha))
                log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Ewald corr.'):
            return compute_ewald_corr_dd(
                self.system.pos, self.system.charges, self.system.dipoles, self.system.cell,
                self.alpha, self.scalings.stab, gpos, vtens
            )

class ForcePartEwaldNeutralizing(ForcePart):
    '''Neutralizing background correction for 3D periodic systems that are
       charged.

       This term is only required of the system is not neutral.
    '''
    def __init__(self, system, alpha, dielectric=1.0):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           **Optional arguments:**

           dielectric
                The scalar relative permittivity of the system.
        '''
        ForcePart.__init__(self, 'ewald_neut', system)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.dielectric = dielectric
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()
                log('  alpha:                   %s' % log.invlength(self.alpha))
                log('  relative permittivity:   %5.3f' % self.dielectric)
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Ewald neut.'):
            #TODO: interaction of dipoles with background? I think this is zero, need proof...
            fac = self.system.charges.sum()**2*np.pi/(2.0*self.system.cell.volume*self.alpha**2)/self.dielectric
            if self.system.radii is not None:
                fac -= self.system.charges.sum()*np.pi/(2.0*self.system.cell.volume)*np.sum( self.system.charges*self.system.radii**2 )/self.dielectric
            if vtens is not None:
                vtens.ravel()[::4] -= fac
            return fac


class ForcePartValence(ForcePart):
    '''The covalent part of a force-field model.

       The covalent force field is implemented in a three-layer approach,
       similar to the implementation of a neural network:

       1. The first layer consists of a :class:`yaff.pes.dlist.DeltaList` object
          that computes all the relative vectors needed for the internal
          coordinates in the covalent energy terms. This list is automatically
          built up as energy terms are added with the ``add_term`` method. This
          list also takes care of transforming `derivatives of the energy
          towards relative vectors` into `derivatives of the energy towards
          Cartesian coordinates and the virial tensor`.

       2. The second layer consist of a
          :class:`yaff.pes.iclist.InternalCoordinateList` object that computes
          the internal coordinates, based on the ``DeltaList``. This list is
          also automatically built up as energy terms are added. The same list
          is also responsible for transforming `derivatives of the energy
          towards internal coordinates` into `derivatives of the energy towards
          relative vectors`.

       3. The third layers consists of a :class:`yaff.pes.vlist.ValenceList`
          object. This list computes the covalent energy terms, based on the
          result in the ``InternalCoordinateList``. This list also computes the
          derivatives of the energy terms towards the internal coordinates.

       The computation of the covalent energy is the so-called `forward code
       path`, which consists of running through steps 1, 2 and 3, in that order.
       The derivatives of the energy are computed in the so-called `backward
       code path`, which consists of taking steps 1, 2 and 3 in reverse order.
       This basic idea of back-propagation for the computation of derivatives
       comes from the field of neural networks. More details can be found in the
       chapter, :ref:`dg_sec_backprop`.
    '''
    def __init__(self, system):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.
        '''
        ForcePart.__init__(self, 'valence', system)
        self.dlist = DeltaList(system)
        self.iclist = InternalCoordinateList(self.dlist)
        self.vlist = ValenceList(self.iclist)
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()

    def add_term(self, term):
        '''Add a new term to the covalent force field.

           **Arguments:**

           term
                An instance of the class :class:`yaff.pes.ff.vlist.ValenceTerm`.

           In principle, one should add all energy terms before calling the
           ``compute`` method, but with the current implementation of Yaff,
           energy terms can be added at any time. (This may change in future.)
        '''
        if log.do_high:
            with log.section('VTERM'):
                log('%7i&%s %s' % (self.vlist.nv, term.get_log(), ' '.join(ic.get_log() for ic in term.ics)))
        self.vlist.add_term(term)

    def _internal_compute(self, gpos, vtens):
        with timer.section('Valence'):
            self.dlist.forward()
            self.iclist.forward()
            energy = self.vlist.forward()
            if not ((gpos is None) and (vtens is None)):
                self.vlist.back()
                self.iclist.back()
                self.dlist.back(gpos, vtens)
            return energy


class ForcePartPressure(ForcePart):
    '''Applies a constant istropic pressure.'''
    def __init__(self, system, pext):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           pext
                The external pressure. (Positive will shrink the system.) In
                case of 2D-PBC, this is the surface tension. In case of 1D, this
                is the linear strain.

           This force part is only applicable to systems that are periodic.
        '''
        if system.cell.nvec == 0:
            raise ValueError('The system must be periodic in order to apply a pressure')
        ForcePart.__init__(self, 'press', system)
        self.system = system
        self.pext = pext
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Valence'):
            cell = self.system.cell
            if (vtens is not None):
                rvecs = cell.rvecs
                if cell.nvec == 1:
                    vtens += self.pext/cell.volume*np.outer(rvecs[0], rvecs[0])
                elif cell.nvec == 2:
                    vtens += self.pext/cell.volume*(
                          np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        + np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        - np.dot(rvecs[1], rvecs[0])*np.outer(rvecs[0], rvecs[1])
                        - np.dot(rvecs[0], rvecs[1])*np.outer(rvecs[1], rvecs[0])
                    )
                elif cell.nvec == 3:
                    gvecs = cell.gvecs
                    vtens += self.pext*cell.volume*np.identity(3)
                else:
                    raise NotImplementedError
            return cell.volume*self.pext


class ForcePartGrid(ForcePart):
    '''Energies obtained by grid interpolation.'''
    def __init__(self, system, grids):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           grids
                A dictionary with (ffatype, grid) items. Each grid must be a
                three-dimensional array with energies.

           This force part is only applicable to systems that are 3D periodic.
        '''
        if system.cell.nvec != 3:
            raise ValueError('The system must be 3d periodic for the grid term.')
        for grid in grids.itervalues():
            if grid.ndim != 3:
                raise ValueError('The energy grids must be 3D numpy arrays.')
        ForcePart.__init__(self, 'grid', system)
        self.system = system
        self.grids = grids
        if log.do_medium:
            with log.section('FPINIT'):
                log('Force part: %s' % self.name)
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section('Grid'):
            if gpos is not None:
                raise NotImplementedError('Cartesian gradients are not supported yet in ForcePartGrid')
            if vtens is not None:
                raise NotImplementedError('Cell deformation are not supported by ForcePartGrid')
            cell = self.system.cell
            result = 0
            for i in xrange(self.system.natom):
                grid = self.grids[self.system.get_ffatype(i)]
                result += compute_grid3d(self.system.pos[i], cell, grid)
            return result
