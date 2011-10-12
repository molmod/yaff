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
from yaff.pes.ext import Cell


__all__ = ['System']


def check_name(name):
    """Raise an error if the given name is invalid."""
    for symbol in ':%=<>@()&|!':
        if symbol in name:
            raise ValueError('A scope or atom type name should not contain \'%s\'. Invalid name: \'%s\'' % (symbol, name))
    if symbol[0].isdigit():
        raise ValueError('A scope or atom type name should not start with a digit.')


class System(object):
    def __init__(self, numbers, pos, ffatypes=None, ffatype_ids=None, scopes=None, scope_ids=None, bonds=None, rvecs=None, charges=None, masses=None):
        '''
           **Arguments:**

           numbers
                A numpy array with atomic numbers

           pos
                A numpy array (N,3) with atomic coordinates in bohr.

           **Optional arguments:**

           ffatypes
                A list of labels of the force field atom types.

           ffatype_ids
                A list of atom type indexes that links each atom with an element
                of the list ffatypes. If this argument is not present, while
                ffatypes is given, it is assumed that ffatypes contains an
                atom type for every element, i.e. that it is a list with length
                natom. In that case, it will be converted automatically to
                a short ffatypes list with only unique elements (within each
                scope) together with a corresponding ffatype_ids array.

           scopes
                A list with scope names

           scope_ids
                A list of scope indexes that links each atom with an element of
                the scopes list. If this argument is not present, while scopes
                is given, it is assumed that scopes contains a scope name for
                every atom, i.e. that it is a list with length natom. In that
                case, it will be converted automatically to a scopes list
                with only unique name together with a corresponding scope_ids
                array.

           bonds
                a numpy array (B,2) with atom indexes (counting starts from
                zero) to define the chemical bonds.

           rvecs
                An array whose rows are the unit cell vectors. At most three
                rows are allowed, each containg three Cartesian coordinates.

           charges
                An array of atomic charges

           masses
                The atomic masses (in atomic units, i.e. m_e)


           Several attributes are derived from the (optional) arguments:

           * ``cell`` contains the rvecs attribute and is an instance of the
             ``Cell`` class.

           * ``neighs1``, ``neighs2`` and ``neighs3`` are dictionaries derived
             from ``bonds`` that contain atoms that are separated 1, 2 and 3
             bonds from a given atom, respectively. This means that i in
             system.neighs3[j] is ``True`` if there are three bonds between
             atoms i and j.
        '''
        if len(numbers.shape) != 1:
            raise ValueError('Argument numbers must be a one-dimensional array.')
        if pos.shape != (len(numbers), 3):
            raise ValueError('The pos array must have Nx3 rows. Mismatch with numbers argument, which myst have shape (N,).')
        if len(ffatypes) != len(numbers):
            raise ValueError('The size of numbers and ffatypes does not match.')
        self.numbers = numbers
        self.pos = pos
        self.ffatypes = ffatypes
        self.ffatype_ids = ffatype_ids
        self.scopes = scopes
        self.scope_ids = scope_ids
        self.bonds = bonds
        self.cell = Cell(rvecs)
        self.charges = charges
        self.masses = masses
        # compute some derived attributes
        self._init_derived()

    def _init_derived(self):
        if self.bonds is not None:
            self._init_derived_bonds()
        if self.scopes is not None:
            self.__init_derived_scopes()
        elif self.scope_ids is not None:
            raise ValueError('The scope_ids only make sense when the scopes argument is given.')
        if self.ffatypes is not None:
            self.__init_derived_ffatypes()
        elif self.ffatype_ids is not None:
            raise ValueError('The ffatype_ids only make sense when the ffatypes argument is given.')

    def _init_derived_bonds(self):
        # 1-bond neighbors
        self.neighs1 = dict((i,set([])) for i in xrange(self.natom))
        for i0, i1 in self.bonds:
            self.neighs1[i0].add(i1)
            self.neighs1[i1].add(i0)
        # 2-bond neighbors
        self.neighs2 = dict((i,set([])) for i in xrange(self.natom))
        for i0, n0 in self.neighs1.iteritems():
            for i1 in n0:
                for i2 in self.neighs1[i1]:
                    # Require that there are no shorter paths than two bonds between
                    # i0 and i2. Also avoid duplicates.
                    if i2 > i0 and i2 not in self.neighs1[i0]:
                        self.neighs2[i0].add(i2)
                        self.neighs2[i2].add(i0)
        # 3-bond neighbors
        self.neighs3 = dict((i,set([])) for i in xrange(self.natom))
        for i0, n0 in self.neighs1.iteritems():
            for i1 in n0:
                for i3 in self.neighs2[i1]:
                    # Require that there are no shorter paths than three bonds
                    # between i0 and i3. Also avoid duplicates.
                    if i3 != i0 and i3 not in self.neighs1[i0] and i3 not in self.neighs2[i0]:
                        self.neighs3[i0].add(i3)
                        self.neighs3[i3].add(i0)

    def _init_derived_scopes(self):
        if self.scope_ids is None:
            if len(self.scopes) != self.natom:
                raise TypeError('When the scope_ids are derived automatically, the length of the scopes list must match the number of atoms.')
            lookup = {}
            scopes = []
            self.scope_ids = np.zeros(self.natom, int)
            for i in xrange(self.natom):
                scope = self.scopes[i]
                scope_id = lookup.get(scope)
                if scope_id is None:
                    scope_id = len(scopes)
                    scopes.append(scope)
                    lookup[scope] = scope_id
                self.scope_ids[i] = scope_id
            self.scopes = scopes
        for scope in self.scopes:
            check_name(scope)
        # check the range of the ids
        if self.scope_ids.min() < 0 or self.scope_ids.max() > len(self.scope):
            raise ValueError('The ffatype_ids our out of bounds.')

    def _init_derived_ffatypes(self):
        if self.ffatype_ids is None:
            if len(self.ffatypes) != self.natom:
                raise TypeError('When the ffatype_ids are derived automatically, the length of the ffatypes list must match the number of atoms.')
            lookup = {}
            ffatypes = []
            self.ffatype_ids = np.zeros(self.natom, int)
            for i in xrange(self.natom):
                if self.scope_ids is None:
                    ffatype = self.ffatypes[i]
                    key = ffatype, None
                else:
                    scope_id = self.scope_ids[i]
                    ffatype = self.ffatypes[i]
                    key = ffatype, scope_id
                ffatype_id = lookup.get(key)[0]
                if ffatype_id is None:
                    ffatype_id = len(ffatypes)
                    ffatypes.append(ffatypes)
                    lookup[key] = ffatypes_id
                self.ffatypes_ids[i] = ffatypes_id
            self.ffatypes = ffatypes
        for ffatype in self.ffatypes:
            check_name(ffatype)
        # check the range of the ids
        if self.ffatype_ids.min() < 0 or self.ffatype_ids.max() > len(self.ffatypes):
            raise ValueError('The ffatype_ids our out of bounds.')
        # check that each ffatype_id occurs in only one scope_id
        if self.scopes is not None:
            raise NotImplementedError

    natom = property(lambda self: len(self.pos))

    @classmethod
    def from_file(cls, *fns, **user_kwargs):
        """Load a system from one or more files

           **Arguments:**

           fn1, fn2, ...
                A list if filenames that are read in order. Information in later
                files overrides information in earlier files.

           **Optional arguments:**

           Any argument from the constructure. These must be given with
           keywords.

           **Supported file formats**

           .xyz
                Standard Cartesian coordinates file (in angstroms). Atomic
                positions and atomic numbers are read from this file. If the
                title consists of 3, 6 or 9 numbers, each group of three numbers
                is interpreted as a cell vector (in angstroms). A guess of the
                bonds will be made based on inter-atomic distances.

           .psf
                Atom types and bonds are read from this file

           .chk
                Internal text-based checkpoint format. It just contains a
                dictionary with the constructor arguments.
        """
        log.enter('SYS')
        kwargs = {}
        for fn in fns:
            if fn.endswith('.xyz'):
                from molmod import Molecule
                mol = Molecule.from_file(fn)
                kwargs['numbers'] = mol.numbers
                kwargs['pos'] = mol.coordinates
                words = mol.title.split()
                if len(words) == 9:
                    try:
                        rvecs = np.array([float(w) for w in words]).reshape((3,-1))*angstrom
                        kwargs['rvecs'] = rvecs
                    except ValueError:
                        rvecs = None
                    if rvecs is not None:
                        mol.unit_cell = UnitCell(rvecs.transpose())
                mol.set_default_graph()
                if len(mol.graph.edges) > 0:
                    kwargs['bonds'] = np.array(mol.graph.edges)
            elif fn.endswith('.psf'):
                from molmod.io import PSFFile
                psf = PSFFile(fn)
                kwargs['ffatypes'] = psf.atom_types
                kwargs['bonds'] = np.array(psf.bonds, copy=False)
                kwargs['charges'] = np.array(psf.charges, copy=False)
            elif fn.endswith('.chk'):
                from molmod.io import load_chk
                kwargs.update(load_chk(fn))
            else:
                raise IOError('Can not read from file \'%s\'.' % fn)
            if log.do_high:
                log('Read system parameters from %s.' % fn)
        kwargs.update(user_kwargs)
        log.leave()
        return cls(**kwargs)

    def set_standard_masses(self):
        log.enter('SYS')
        from molmod.periodic import periodic
        if self.masses is not None:
            if log.do_warning:
                log.warn('Overwriting existing masses with default masses.')
        self.masses = np.array([periodic[n].mass for n in self.numbers])
        log.leave()

    def to_file(self, fn_chk):
        """Write the system in the internal checkpoint format.

           **Arguments:**

           fn_chk
                The file to write to.

           All data are stored in atomic units.
        """
        from molmod.io import dump_chk
        dump_chk(fn_chk, {
            'numbers': self.numbers,
            'pos': self.pos,
            'ffatypes': self.ffatypes,
            'ffatype_ids': self.ffatype_ds,
            'scopes': self.scopes,
            'scope_ids': self.scope_ds,
            'bonds': self.bonds,
            'rvecs': self.cell.rvecs,
            'charges': self.charges,
            'masses': self.masses,
        })
        if log.do_high:
            log.enter('SYS')
            log('Wrote system to %s.' % fn_chk)
            log.leave()
