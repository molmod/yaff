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

from yaff.log import log
from yaff.atselect import check_name, atsel_compile
from yaff.pes.ext import Cell


__all__ = ['System']


def _unravel_triangular(i):
    """Transform a flattened triangular matrix index to row and column indexes

       It is assumed that the diagonal elements are not included in the
       flattened triangular matrix.
    """
    i0 = int(np.floor(0.5*(np.sqrt(1+8*i)-1)))+1
    i1 = i - (i0*(i0-1))/2
    return i0, i1


class System(object):
    def __init__(self, numbers, pos, scopes=None, scope_ids=None, ffatypes=None, ffatype_ids=None, bonds=None, rvecs=None, charges=None, masses=None):
        '''
           **Arguments:**

           numbers
                A numpy array with atomic numbers

           pos
                A numpy array (N,3) with atomic coordinates in Bohr.

           **Optional arguments:**

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

           bonds
                a numpy array (B,2) with atom indexes (counting starts from
                zero) to define the chemical bonds.

           rvecs
                An array whose rows are the unit cell vectors. At most three
                rows are allowed, each containing three Cartesian coordinates.

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
            raise ValueError('The pos array must have Nx3 rows. Mismatch with numbers argument with shape (N,).')
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
        with log.section('SYS'):
            self._init_derived()

    def _init_derived(self):
        if self.bonds is not None:
            self._init_derived_bonds()
        if self.scopes is not None:
            self._init_derived_scopes()
        elif self.scope_ids is not None:
            raise ValueError('The scope_ids only make sense when the scopes argument is given.')
        if self.ffatypes is not None:
            self._init_derived_ffatypes()
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
        if self.scope_ids.min() != 0 or self.scope_ids.max() != len(self.scopes)-1:
            raise ValueError('The ffatype_ids have incorrect bounds.')
        if log.do_medium:
            log('The following scopes are present in the system:')
            log.hline()
            log('                 Scope   ID   Number of atoms')
            log.hline()
            for scope_id, scope in enumerate(self.scopes):
                log('%22s  %3i       %3i' % (scope, scope_id, (self.scope_ids==scope_id).sum()))
            log.hline()
            log.blank()

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
                ffatype_id = lookup.get(key)
                if ffatype_id is None:
                    ffatype_id = len(ffatypes)
                    ffatypes.append(ffatype)
                    lookup[key] = ffatype_id
                self.ffatype_ids[i] = ffatype_id
            self.ffatypes = ffatypes
        for ffatype in self.ffatypes:
            check_name(ffatype)
        # check the range of the ids
        if self.ffatype_ids.min() != 0 or self.ffatype_ids.max() != len(self.ffatypes)-1:
            raise ValueError('The ffatype_ids have incorrect bounds.')
        # differentiate ffatype_ids if the same ffatype_id is used in different
        # scopes
        if self.scopes is not None:
            self.ffatype_id_to_scope_id = {}
            fixed_fids = {}
            for i in xrange(self.natom):
                fid = self.ffatype_ids[i]
                sid = self.ffatype_id_to_scope_id.get(fid)
                if sid is None:
                    self.ffatype_id_to_scope_id[fid] = self.scope_ids[i]
                elif sid != self.scope_ids[i]:
                    # We found the same ffatype_id in a different scope_id. This
                    # must be fixed. First check if we have already a new
                    # scope_id ready
                    sid = self.scope_ids[i]
                    new_fid = fixed_fids.get((sid, fid))
                    if new_fid is None:
                        # No previous new fid create, do it now.
                        new_fid = len(self.ffatypes)
                        # Copy the ffatype label
                        self.ffatypes.append(self.ffatypes[fid])
                        # Keep track of the new fid
                        fixed_fids[(sid, fid)] = new_fid
                        if log.do_warning:
                            log.warn('Atoms with type ID %i in scope %s were changed to type ID %i.' % (fid, self.scopes[sid], new_fid))
                    # Apply the new fid
                    self.ffatype_ids[i] = new_fid
                    self.ffatype_id_to_scope_id[new_fid] = sid
        # Turn the ffatypes in the scopes into array
        if self.ffatypes is not None:
            self.ffatypes = np.array(self.ffatypes, copy=False)
        if self.scopes is not None:
            self.scopes = np.array(self.scopes, copy=False)
        # check the range of the ids
        if self.ffatype_ids.min() != 0 or self.ffatype_ids.max() != len(self.ffatypes)-1:
            raise ValueError('The ffatype_ids have incorrect bounds.')
        if log.do_medium:
            log('The following atom types are present in the system:')
            log.hline()
            if self.scopes is None:
                log('             Atom type   ID   Number of atoms')
                log.hline()
                for ffatype_id, ffatype in enumerate(self.ffatypes):
                    log('%22s  %3i       %3i' % (ffatype, ffatype_id, (self.ffatype_ids==ffatype_id).sum()))
            else:
                log('                 Scope              Atom type   ID   Number of atoms')
                log.hline()
                for ffatype_id, ffatype in enumerate(self.ffatypes):
                    scope = self.scopes[self.ffatype_id_to_scope_id[ffatype_id]]
                    log('%22s %22s  %3i       %3i' % (scope, ffatype, ffatype_id, (self.ffatype_ids==ffatype_id).sum()))
            log.hline()
            log.blank()
        # TODO: Add double check such that the same atom type implies the same
        # atomic number.

    def get_natom(self):
        """The number of atoms"""
        return len(self.pos)

    natom = property(get_natom)

    def get_nffatype(self):
        """The number of atom types"""
        return len(self.ffatypes)

    nffatype = property(get_nffatype)

    def get_nbond(self):
        '''The number of bonds'''
        if self.bonds is None:
            return 0
        else:
            return len(self.bonds)

    nbond = property(get_nbond)

    @classmethod
    def from_file(cls, *fns, **user_kwargs):
        """Construct a new System instance from one or more files

           **Arguments:**

           fn1, fn2, ...
                A list if filenames that are read in order. Information in later
                files overrides information in earlier files.

           **Optional arguments:**

           Any argument from the default constructor ``__init__``. These must be
           given with keywords.

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
        with log.section('SYS'):
            kwargs = {}
            for fn in fns:
                if fn.endswith('.xyz'):
                    from molmod import Molecule
                    mol = Molecule.from_file(fn)
                    kwargs['numbers'] = mol.numbers.copy()
                    kwargs['pos'] = mol.coordinates.copy()
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
        return cls(**kwargs)

    def get_scope(self, index):
        """Return the of the scope (string) of atom with given index"""
        return self.scopes[self.scope_ids[index]]

    def get_ffatype(self, index):
        """Return the of the ffatype (string) of atom with given index"""
        return self.ffatypes[self.ffatype_ids[index]]

    def get_indexes(self, rule):
        """Return the atom indexes that match the filter ``rule``

           ``rule`` can be a function that accepts two arguments: system and an
           atom index and that returns True of the atom with index i is of a
           given type. On the other hand ``rule`` can be an ATSELECT string that
           defines the atoms of interest.

           A list of atom indexes is returned.
        """
        if isinstance(rule, basestring):
            rule = atsel_compile(rule)
        return np.array([i for i in xrange(self.natom) if rule(self, i)])

    def iter_bonds(self):
        """Iterate over all bonds."""
        for i1, i2 in self.bonds:
            yield i1, i2

    def iter_angles(self):
        """Iterative over all possible valence angles.

           This routine is based on the attribute ``bonds``.
        """
        for i1 in xrange(self.natom):
            for i0 in self.neighs1[i1]:
                for i2 in self.neighs1[i1]:
                    if i0 > i2:
                        yield i0, i1, i2

    def iter_dihedrals(self):
        """Iterative over all possible dihedral angles.

           This routine is based on the attribute ``bonds``.
        """
        for i1, i2 in self.bonds:
            for i0 in self.neighs1[i1]:
                if i0==i2: continue
                for i3 in self.neighs1[i2]:
                    if i1==i3: continue
                    if i0==i3: continue
                    yield i0, i1, i2, i3

    def detect_bonds(self):
        """Initialize the ``bonds`` attribute based on inter-atomic distances

           For each pair of elements, a distance threshold is used to detect
           bonded atoms. The distance threshold is based on a database of known
           bond lengths. If the database does not contain a record for the given
           element pair, the threshold is based on the sum of covalent radii.
        """
        with log.section('SYS'):
            from molmod.bonds import bonds
            if self.bonds is not None:
                if log.do_warning:
                    log.warn('Overwriting existing bonds.')
            work = np.zeros((self.natom*(self.natom-1))/2, float)
            self.cell.compute_distances(work, self.pos)
            ishort = (work < bonds.max_length*1.01).nonzero()[0]
            new_bonds = []
            for i in ishort:
                i0, i1 = _unravel_triangular(i)
                if bonds.bonded(self.numbers[i0], self.numbers[i1], work[i]):
                    new_bonds.append((i0, i1))
            self.bonds = np.array(new_bonds)
            self._init_derived_bonds()

    def detect_ffatypes(self, rules):
        """Initialize the ``ffatypes`` attribute based on ATSELECT rules.

           **Argument:**

           rules
                A list of (ffatype, rule) pairs that will be used to initialize
                the attributes ``self.ffatypes`` and ``self.ffatype_ids``.

           If the system already has FF atom types, they will be overwritten.
        """
        with log.section('SYS'):
            # Give warning if needed
            if self.ffatypes is not None:
                if log.do_warning:
                    log.warn('Overwriting existing FF atom types.')
            # Compile all the rules
            my_rules = []
            for ffatype, rule in rules:
                check_name(ffatype)
                if isinstance(rule, basestring):
                    rule = atsel_compile(rule)
                my_rules.append((ffatype, rule))
            # Use the rules to detect the atom types
            lookup = {}
            self.ffatypes = []
            self.ffatype_ids = np.zeros(self.natom, int)
            for i in xrange(self.natom):
                my_ffatype = None
                for ffatype, rule in my_rules:
                    if rule(self, i):
                        my_ffatype = ffatype
                        break
                if my_ffatype is None:
                    raise ValueError('Could not detect FF atom type of atom %i.' % i)
                ffatype_id = lookup.get(my_ffatype)
                if ffatype_id is None:
                    ffatype_id = len(lookup)
                    self.ffatypes.append(my_ffatype)
                    lookup[my_ffatype] = ffatype_id
                self.ffatype_ids[i] = ffatype_id
            # Make sure all is done well ...
            self._init_derived_ffatypes()

    def set_standard_masses(self):
        """Initialize the ``masses`` attribute based on the atomic numbers."""
        with log.section('SYS'):
            from molmod.periodic import periodic
            if self.masses is not None:
                if log.do_warning:
                    log.warn('Overwriting existing masses with default masses.')
            self.masses = np.array([periodic[n].mass for n in self.numbers])

    def align_cell(self, lcs=None, swap=True):
        """Align the unit cell with respect to the Cartesian Axes frame

           **Optional Arguments:**

           lcs
                The linear combinations of the unit cell that must get aligned.
                This is a 2x3 array, where each row represents a linear
                combination of cell vectors. The first row is for alignment with
                the x-axis, second for the z-axis. The default value is::

                    np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                    ])

           swap
                By default, the first alignment is done with the z-axis, then
                with the x-axis. The order is reversed when swap is set to
                False.

           The alignment of the first linear combination is always perfect. The
           alignment of the second linear combination is restricted to a plane.
           The cell is always made right-handed. The coordinates are also
           rotated with respect to the origin, but never inverted.

           The attributes of the system are modified in-place. Note that this
           method only works on 3D periodic systems.
        """
        from molmod import Rotation, deg
        # define the target
        target = np.array([
            [1, 0, 0],
            [0, 0, 1],
        ])

        # default value for linear combination
        if lcs is None:
            lcs = target.copy()

        # The starting values
        pos = self.pos
        rvecs = self.cell.rvecs.copy()
        if rvecs.shape != (3,3):
            raise TypeError('The align_cell method only supports 3D periodic systems.')

        # Optionally swap a cell vector if the cell is not right-handed.
        if np.linalg.det(rvecs) < 0:
            # Find a reasonable vector to swap...
            index = rvecs.sum(axis=1).argmin()
            rvecs[index] *= -1

        # Define the source
        source = np.dot(lcs, rvecs)

        # Do the swapping
        if swap:
            target = target[::-1]
            source = source[::-1]

        # auxiliary function
        def get_angle_axis(t, s):
            cos = np.dot(s, t)/np.linalg.norm(s)/np.linalg.norm(t)
            angle = np.arccos(np.clip(cos, -1, 1))
            axis = np.cross(s, t)
            return angle, axis

        # first alignment
        angle, axis = get_angle_axis(target[0], source[0])
        if np.linalg.norm(axis) > 0:
            r1 = Rotation.from_properties(angle, axis, False)
            pos = r1*pos
            rvecs = r1*rvecs
            source = r1*source

        # second alignment
        # Make sure the source is orthogonal to target[0]
        s1p = source[1] - target[0]*np.dot(target[0], source[1])
        angle, axis = get_angle_axis(target[1], s1p)
        r2 = Rotation.from_properties(angle, axis, False)
        pos = r2*pos
        rvecs = r2*rvecs

        # assign
        self.pos = pos
        self.cell = Cell(rvecs)

    def supercell(self, *reps):
        """Return a supercell of the system.

           **Arguments:**

           reps
                An array with repetitions, which must have the same number of
                elements as the number of cell vectors.

           If this method is called with a non-periodic system, a TypeError is
           raised.
        """
        if self.cell.nvec == 0:
            raise TypeError('Can not create a supercell of a non-periodic system.')
        if self.cell.nvec != len(reps):
            raise TypeError('The number of repetitions must match the number of cell vectors.')
        if not isinstance(reps, tuple):
            raise TypeError('The reps argument must be a tuple')
        # A dictionary with new arguments for the construction of the supercell
        new_args = {}

        # A) No repetitions
        if self.ffatypes is not None:
            new_args['ffatypes'] = self.ffatypes.copy()
        if self.scopes is not None:
            new_args['scopes'] = self.scopes.copy()

        # B) Simple repetitions
        rep_all = np.product(reps)
        for attrname in 'numbers', 'ffatype_ids', 'scope_ids', 'charges', 'masses':
            value = getattr(self, attrname)
            if value is not None:
                new_args[attrname] = np.tile(value, rep_all)

        # C) Cell vectors
        new_args['rvecs'] = self.cell.rvecs*np.reshape(reps, (3,1))

        # D) Atom positions
        new_pos = np.zeros((self.natom*rep_all, 3), float)
        start = 0
        for iimage in np.ndindex(reps):
            stop = start+self.natom
            new_pos[start:stop] = self.pos + np.dot(iimage, self.cell.rvecs)
            start = stop
        new_args['pos'] = new_pos

        if self.bonds is not None:
            # E) Bonds
            # E.1) A function that translates a set of image indexes and an old atom
            # index into a new atom index
            offsets = {}
            start = 0
            for iimage in np.ndindex(reps):
                offsets[iimage] = start
                start += self.natom
            def to_new_atom_index(iimage, i):
                return offsets[iimage] + i

            # E.2) Construct extended bond information: for each bond, also keep
            # track of periodic image it connects to. Note that this information
            # is implicit in yaff, and derived using the minimum image convention.
            rel_iimage = {}
            for ibond in xrange(len(self.bonds)):
                i0, i1 = self.bonds[ibond]
                delta = self.pos[i0] - self.pos[i1]
                frac = np.dot(self.cell.gvecs, delta)
                rel_iimage[ibond] = np.ceil(frac-0.5)

            # E.3) Create the new bonds
            new_bonds = np.zeros((len(self.bonds)*rep_all,2), int)
            counter = 0
            for iimage0 in np.ndindex(reps):
                for ibond in xrange(len(self.bonds)):
                    i0, i1 = self.bonds[ibond]
                    # Translate i0 to the new index.
                    j0 = to_new_atom_index(iimage0, i0)
                    # Also translate i1 to the new index. This is a bit more tricky.
                    # The difficult case occurs when the bond between i0 and i1
                    # connects different periodic images. In that case, the change
                    # in periodic image must be taken into account.
                    iimage1 = tuple((iimage0[c] + rel_iimage[ibond][c]) % reps[c] for c in xrange(len(reps)))
                    j1 = to_new_atom_index(iimage1, i1)
                    new_bonds[counter,0] = j0
                    new_bonds[counter,1] = j1
                    counter += 1
            new_args['bonds'] = new_bonds

        # Done
        return System(**new_args)

    def to_file(self, fn):
        """Write the system to a file

           **Arguments:**

           fn
                The file to write to.

           Supported formats are:

           chk
                Internal human-readable checkpoint format. This format includes
                all the information of a system object. All data are stored in
                atomic units.

           h5
                Internal binary checkpoint format. This format includes
                all the information of a system object. All data are stored in
                atomic units.

           xyz
                A simple file with atomic positions and elements. Coordinates
                are written in Angstroms.
        """
        #TODO: Add a few common formats like PDB
        # (Cell parameters, connectivity, atom types (4char), scope -> chains)
        if fn.endswith('.chk'):
            from molmod.io import dump_chk
            dump_chk(fn, {
                'numbers': self.numbers,
                'pos': self.pos,
                'ffatypes': self.ffatypes,
                'ffatype_ids': self.ffatype_ids,
                'scopes': self.scopes,
                'scope_ids': self.scope_ids,
                'bonds': self.bonds,
                'rvecs': self.cell.rvecs,
                'charges': self.charges,
                'masses': self.masses,
            })
        elif fn.endswith('.h5'):
            with h5.File(fn, 'w') as f:
                self.to_hdf5(f)
        elif fn.endswith('.xyz'):
            from molmod.io import XYZWriter
            from molmod.periodic import periodic
            xyz_writer = XYZWriter(fn, [periodic[n].symbol for n in self.numbers])
            xyz_writer.dump(str(self), self.pos)
        else:
            raise NotImplementedError('The extension of %s does not correspond to any known format.' % fn)
        if log.do_high:
            with log.section('SYS'):
                log('Wrote system to %s.' % fn)

    def to_hdf5(self, f):
        """Write the system to a HDF5 file.

           **Arguments:**

           f
                A Writable h5.File object.
        """
        if 'system' in f:
            raise ValueError('The HDF5 file already contains a system description.')
        sgrp = f.create_group('system')
        sgrp.create_dataset('numbers', data=self.numbers)
        sgrp.create_dataset('pos', data=self.pos)
        if self.scopes is not None:
            sgrp.create_dataset('scopes', data=self.scopes, dtype='a22')
            sgrp.create_dataset('scope_ids', data=self.scope_ids)
        if self.ffatypes is not None:
            sgrp.create_dataset('ffatypes', data=self.ffatypes, dtype='a22')
            sgrp.create_dataset('ffatype_ids', data=self.ffatype_ids)
        if self.bonds is not None:
            sgrp.create_dataset('bonds', data=self.bonds)
        if self.cell.nvec > 0:
            sgrp.create_dataset('rvecs', data=self.cell.rvecs)
        if self.charges is not None:
            sgrp.create_dataset('charges', data=self.charges)
        if self.masses is not None:
            sgrp.create_dataset('masses', data=self.masses)
