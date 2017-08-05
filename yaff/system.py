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
'''Representation of a molecular systems'''


from __future__ import division

import numpy as np, h5py as h5

from yaff.log import log
from yaff.atselect import check_name, atsel_compile, iter_matches
from yaff.pes.ext import Cell


__all__ = ['System']


def _unravel_triangular(i):
    """Transform a flattened triangular matrix index to row and column indexes

       It is assumed that the diagonal elements are not included in the
       flattened triangular matrix.
    """
    i0 = int(np.floor(0.5*(np.sqrt(1+8*i)-1)))+1
    i1 = i - (i0*(i0-1))//2
    return i0, i1


class System(object):
    def __init__(self, numbers, pos, scopes=None, scope_ids=None, ffatypes=None,
                 ffatype_ids=None, bonds=None, rvecs=None, charges=None,
                 radii=None, valence_charges=None, dipoles=None, radii2=None,
                 masses=None):
        r'''Initialize a System object.

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

           radii
                An array of atomic radii, :math:`R_{A,c}`, that determine shape of the atomic
                charge distribution:

                .. math::

                    \rho_{A,c}(\mathbf{r}) = \frac{q_A}{\pi^{3/2}R_{A,c}^3} \exp\left(
                    -\frac{|r - \mathbf{R}_A|^2}{R_{A,c}^2}
                    \right)

           valence_charges
                In case a point-core + distribute valence charge is used, this
                vector contains the valence charges. The core charges can be
                computed by subtracting the valence charges from the net
                charges.

           dipoles
                An array of atomic dipoles

           radii2
                An array of atomic radii, :math:`R_{A,d}`, that determine shape of the
                atomic dipole distribution:

                .. math::

                   \rho_{A,d}(\mathbf{r}) = -2\frac{\mathbf{d}_A \cdot (\mathbf{r} - \mathbf{R}_A)}{
                   \sqrt{\pi} R_{A,d}^5
                   }\exp\left(
                    -\frac{|r - \mathbf{R}_A|^2}{R_{A,d}^2}
                    \right)

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
        self.radii = radii
        self.valence_charges = valence_charges
        self.dipoles = dipoles
        self.radii2 = radii2
        self.masses = masses
        with log.section('SYS'):
            # report some stuff
            self._init_log()
            # compute some derived attributes
            self._init_derived()

    def _init_log(self):
        if log.do_medium:
            log('Unit cell')
            log.hline()
            log('Number of periodic dimensions: %i' % self.cell.nvec)
            lengths, angles = self.cell.parameters
            names = 'abc'
            for i in range(len(lengths)):
                log('Cell parameter %5s: %10s' % (names[i], log.length(lengths[i])))
            names = 'alpha', 'beta', 'gamma'
            for i in range(len(angles)):
                log('Cell parameter %5s: %10s' % (names[i], log.angle(angles[i])))
            log.hline()
            log.blank()

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
        self.neighs1 = dict((i,set([])) for i in range(self.natom))
        for i0, i1 in self.bonds:
            self.neighs1[i0].add(i1)
            self.neighs1[i1].add(i0)
        # 2-bond neighbors
        self.neighs2 = dict((i,set([])) for i in range(self.natom))
        for i0, n0 in self.neighs1.items():
            for i1 in n0:
                for i2 in self.neighs1[i1]:
                    # Require that there are no shorter paths than two bonds between
                    # i0 and i2. Also avoid duplicates.
                    if i2 > i0 and i2 not in self.neighs1[i0]:
                        self.neighs2[i0].add(i2)
                        self.neighs2[i2].add(i0)
        # 3-bond neighbors
        self.neighs3 = dict((i,set([])) for i in range(self.natom))
        for i0, n0 in self.neighs1.items():
            for i1 in n0:
                for i3 in self.neighs2[i1]:
                    # Require that there are no shorter paths than three bonds
                    # between i0 and i3. Also avoid duplicates.
                    if i3 != i0 and i3 not in self.neighs1[i0] and i3 not in self.neighs2[i0]:
                        self.neighs3[i0].add(i3)
                        self.neighs3[i3].add(i0)
        # 4-bond neighbors
        self.neighs4 = dict((i,set([])) for i in range(self.natom))
        for i0, n0 in self.neighs1.items():
            for i1 in n0:
                for i4 in self.neighs3[i1]:
                    # Require that there are no shorter paths than three bonds
                    # between i0 and i4. Also avoid duplicates.
                    if i4 != i0 and i4 not in self.neighs1[i0] and i4 not in self.neighs2[i0] and i4 not in self.neighs3[i0]:
                        self.neighs4[i0].add(i4)
                        self.neighs4[i4].add(i0)
        # report some basic stuff on screen
        if log.do_medium:
            log('Analysis of the bonds:')
            bond_types = {}
            for i0, i1 in self.bonds:
                key = tuple(sorted([self.numbers[i0], self.numbers[i1]]))
                bond_types[key] = bond_types.get(key, 0) + 1
            log.hline()
            log(' First   Second   Count')
            for (num0, num1), count in sorted(bond_types.items()):
                log('%6i   %6i   %5i' % (num0, num1, count))
            log.hline()
            log.blank()

            log('Analysis of the neighbors:')
            log.hline()
            log('Number of first neighbors:  %6i' % (sum(len(n) for n in self.neighs1.values())//2))
            log('Number of second neighbors: %6i' % (sum(len(n) for n in self.neighs2.values())//2))
            log('Number of third neighbors:  %6i' % (sum(len(n) for n in self.neighs3.values())//2))
            # Collect all types of 'environments' for each element. This is
            # useful to double check the bonds
            envs = {}
            for i0 in range(self.natom):
                num0 = self.numbers[i0]
                nnums = tuple(sorted(self.numbers[i1] for i1 in self.neighs1[i0]))
                key = (num0, nnums)
                envs[key] = envs.get(key, 0)+1
            # Print the environments on screen
            log.hline()
            log('Element   Neighboring elements   Count')
            for (num0, nnums), count in sorted(envs.items()):
                log('%7i   %20s   %5i' % (num0, ','.join(str(num1) for num1 in nnums), count))
            log.hline()
            log.blank()


    def _init_derived_scopes(self):
        if self.scope_ids is None:
            if len(self.scopes) != self.natom:
                raise TypeError('When the scope_ids are derived automatically, the length of the scopes list must match the number of atoms.')
            lookup = {}
            scopes = []
            self.scope_ids = np.zeros(self.natom, int)
            for i in range(self.natom):
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
            for i in range(self.natom):
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
            for i in range(self.natom):
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

    def _get_natom(self):
        """The number of atoms"""
        return len(self.pos)

    natom = property(_get_natom)

    def _get_nffatype(self):
        """The number of atom types"""
        return len(self.ffatypes)

    nffatype = property(_get_nffatype)

    def _get_nbond(self):
        '''The number of bonds'''
        if self.bonds is None:
            return 0
        else:
            return len(self.bonds)

    nbond = property(_get_nbond)

    @classmethod
    def from_file(cls, *fns, **user_kwargs):
        """Construct a new System instance from one or more files

           **Arguments:**

           fn1, fn2, ...
                A list of filenames that are read in order. Information in later
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
                    allowed_keys = [
                        'numbers', 'pos', 'scopes', 'scope_ids', 'ffatypes',
                        'ffatype_ids', 'bonds', 'rvecs', 'charges', 'radii',
                        'valence_charges', 'dipoles', 'radii2', 'masses',
                    ]
                    for key, value in load_chk(fn).items():
                        if key in allowed_keys:
                            kwargs.update({key: value})
                elif fn.endswith('.h5'):
                    with h5.File(fn, 'r') as f:
                        return cls.from_hdf5(f)
                else:
                    raise IOError('Can not read from file \'%s\'.' % fn)
                if log.do_high:
                    log('Read system parameters from %s.' % fn)
            kwargs.update(user_kwargs)
        return cls(**kwargs)

    @classmethod
    def from_hdf5(cls, f):
        '''Create a system from an HDF5 file/group containing a system group

           **Arguments:**

           f
                An open h5.File object with a system group. The system group
                must at least contain a numbers and pos dataset.
        '''
        sgrp = f['system']
        kwargs = {
            'numbers': sgrp['numbers'][:],
            'pos': sgrp['pos'][:],
        }
        for key in 'scope_ids', 'ffatype_ids', 'bonds', 'rvecs', 'charges', 'masses':
            if key in sgrp:
                kwargs[key] = sgrp[key][:]
        # String arrays have to be converted back to unicode...
        for key in 'scopes', 'ffatypes':
            if key in sgrp:
                kwargs[key] = sgrp[key][:].astype('U')
        if log.do_high:
            log('Read system parameters from %s.' % f.filename)
        return cls(**kwargs)

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
            # Strings have to be stored as ascii
            sgrp.create_dataset('scopes', data=self.scopes.astype('S22'))
            sgrp.create_dataset('scope_ids', data=self.scope_ids)
        if self.ffatypes is not None:
            # Strings have to be stored as ascii
            sgrp.create_dataset('ffatypes', data=self.ffatypes.astype('S22'))
            sgrp.create_dataset('ffatype_ids', data=self.ffatype_ids)
        if self.bonds is not None:
            sgrp.create_dataset('bonds', data=self.bonds)
        if self.cell.nvec > 0:
            sgrp.create_dataset('rvecs', data=self.cell.rvecs)
        if self.charges is not None:
            sgrp.create_dataset('charges', data=self.charges)
        if self.masses is not None:
            sgrp.create_dataset('masses', data=self.masses)


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
        if isinstance(rule, str):
            rule = atsel_compile(rule)
        return np.array([i for i in range(self.natom) if rule(self, i)])

    def iter_bonds(self):
        """Iterate over all bonds."""
        if self.bonds is not None:
            for i1, i2 in self.bonds:
                yield i1, i2

    def iter_angles(self):
        """Iterative over all possible valence angles.

           This routine is based on the attribute ``bonds``.
        """
        if self.bonds is not None:
            for i1 in range(self.natom):
                for i0 in self.neighs1[i1]:
                    for i2 in self.neighs1[i1]:
                        if i0 > i2:
                            yield i0, i1, i2

    def iter_dihedrals(self):
        """Iterative over all possible dihedral angles.

           This routine is based on the attribute ``bonds``.
        """
        if self.bonds is not None:
            for i1, i2 in self.bonds:
                for i0 in self.neighs1[i1]:
                    if i0==i2: continue
                    for i3 in self.neighs1[i2]:
                        if i1==i3: continue
                        if i0==i3: continue
                        yield i0, i1, i2, i3

    def iter_oops(self):
        """Iterative over all possible oop patterns."

           This routine is based on the attribute ``bonds``.
        """
        if self.bonds is not None:
            for i3 in range(self.natom):
                if len(self.neighs1[i3])==3:
                    i0, i1, i2 = self.neighs1[i3]
                    yield i0, i1, i2, i3

    def detect_bonds(self, exceptions=None):
        """Initialize the ``bonds`` attribute based on inter-atomic distances

           **Optional argument:**

           exceptions:
                Specify custom threshold for certain pairs of elements. This
                must be a dictionary with ((num0, num1), threshold) as items.

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
            work = np.zeros((self.natom*(self.natom-1))//2, float)
            self.cell.compute_distances(work, self.pos)
            ishort = (work < bonds.max_length*1.01).nonzero()[0]
            new_bonds = []
            for i in ishort:
                i0, i1 = _unravel_triangular(i)
                n0 = self.numbers[i0]
                n1 = self.numbers[i1]
                if exceptions is not None:
                    threshold = exceptions.get((n0, n1))
                    if threshold is None and n0!=n1:
                        threshold = exceptions.get((n1, n0))
                    if threshold is not None:
                        if work[i] < threshold:
                            new_bonds.append([i0, i1])
                        continue
                if bonds.bonded(n0, n1, work[i]):
                    new_bonds.append([i0, i1])
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
                if isinstance(rule, str):
                    rule = atsel_compile(rule)
                my_rules.append((ffatype, rule))
            # Use the rules to detect the atom types
            lookup = {}
            self.ffatypes = []
            self.ffatype_ids = np.zeros(self.natom, int)
            for i in range(self.natom):
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
        for attrname in 'numbers', 'ffatype_ids', 'scope_ids', 'charges', \
                        'radii', 'valence_charges', 'radii2', 'masses':
            value = getattr(self, attrname)
            if value is not None:
                new_args[attrname] = np.tile(value, rep_all)
        attrname = 'dipoles'
        value = getattr(self, attrname)
        if value is not None:
            new_args[attrname] = np.tile(value, (rep_all, 1))

        # C) Cell vectors
        new_args['rvecs'] = self.cell.rvecs*np.array(reps)[:,None]

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
            for ibond in range(len(self.bonds)):
                i0, i1 = self.bonds[ibond]
                delta = self.pos[i0] - self.pos[i1]
                frac = np.dot(self.cell.gvecs, delta)
                rel_iimage[ibond] = np.ceil(frac-0.5)

            # E.3) Create the new bonds
            new_bonds = np.zeros((len(self.bonds)*rep_all,2), int)
            counter = 0
            for iimage0 in np.ndindex(reps):
                for ibond in range(len(self.bonds)):
                    i0, i1 = self.bonds[ibond]
                    # Translate i0 to the new index.
                    j0 = to_new_atom_index(iimage0, i0)
                    # Also translate i1 to the new index. This is a bit more tricky.
                    # The difficult case occurs when the bond between i0 and i1
                    # connects different periodic images. In that case, the change
                    # in periodic image must be taken into account.
                    iimage1 = tuple((iimage0[c] + rel_iimage[ibond][c]) % reps[c] for c in range(len(reps)))
                    j1 = to_new_atom_index(iimage1, i1)
                    new_bonds[counter,0] = j0
                    new_bonds[counter,1] = j1
                    counter += 1
            new_args['bonds'] = new_bonds

        # Done
        return System(**new_args)

    def remove_duplicate(self, threshold=0.1):
        '''Return a system object in which the duplicate atoms and bonds are removed.

           **Optional argument:**

           threshold
                The minimum distance between two atoms that are supposed to be
                different.

           When it makes sense, properties of overlapping atoms are averaged
           out. In other cases, the atom with the lowest index in a cluster of
           overlapping atoms defines the new value of a property.
        '''
        # compute distances
        ndist = (self.natom*(self.natom-1))//2
        if ndist == 0: # single atom systems, go home ...
            return
        dists = np.zeros(ndist)
        self.cell.compute_distances(dists, self.pos)

        # find clusters of overlapping atoms
        from molmod import ClusterFactory
        cf = ClusterFactory()
        counter = 0
        for i0 in range(self.natom):
            for i1 in range(i0):
                if dists[counter] < threshold:
                    cf.add_related(i0, i1)
                counter += 1
        clusters = [c.items for c in cf.get_clusters()]

        # make a mapping from new to old atoms
        newold = {}
        oldnew = {}
        counter = 0
        for cluster in clusters: # all merged atoms come first
            newold[counter] = sorted(cluster)
            for item in cluster:
                oldnew[item] = counter
            counter += 1
        if len(clusters) > 0:
            old_reduced = set.union(*clusters)
        else:
            old_reduced = []
        for item in range(self.natom): # all remaining atoms follow
            if item not in old_reduced:
                newold[counter] = [item]
                oldnew[item] = counter
                counter += 1
        natom = len(newold)

        def reduce_int_array(old):
            if old is None:
                return None
            else:
                new = np.zeros(natom, old.dtype)
                for inew, iolds in newold.items():
                    new[inew] = old[iolds[0]]
                return new

        def reduce_float_array(old):
            if old is None:
                return None
            else:
                new = np.zeros(natom, old.dtype)
                for inew, iolds in newold.items():
                    new[inew] = old[iolds].mean()
                return new

        def reduce_float_matrix(old):
            '''Reduce array with dim=2'''
            if old is None:
                return None
            else:
                new = np.zeros((natom,np.shape(old)[1]), old.dtype)
                for inew, iolds in newold.items():
                    new[inew] = old[iolds].mean(axis=0)
                return new

        # trivial cases
        numbers = reduce_int_array(self.numbers)
        scope_ids = reduce_int_array(self.scope_ids)
        ffatype_ids = reduce_int_array(self.ffatype_ids)
        charges = reduce_float_array(self.charges)
        radii = reduce_float_array(self.radii)
        valence_charges = reduce_float_array(self.valence_charges)
        dipoles = reduce_float_matrix(self.dipoles)
        radii2 = reduce_float_array(self.radii2)
        masses = reduce_float_array(self.masses)

        # create averaged positions
        pos = np.zeros((natom, 3), float)
        for inew, iolds in newold.items():
            # move to the same image
            oldposs = self.pos[iolds].copy()
            assert oldposs.ndim == 2
            ref = oldposs[0]
            for oldpos in oldposs[1:]:
                delta = oldpos-ref
                self.cell.mic(delta)
                oldpos[:] = delta+ref
            # compute mean position
            pos[inew] = oldposs.mean(axis=0)

        # create reduced list of bonds
        if self.bonds is None:
            bonds = None
        else:
            bonds = set((oldnew[ia], oldnew[ib]) for ia, ib in self.bonds)
            bonds = np.array([bond for bond in bonds])

        return self.__class__(numbers, pos, self.scopes, scope_ids, self.ffatypes, ffatype_ids, bonds, self.cell.rvecs, charges, radii, valence_charges, dipoles, radii2, masses)

    def subsystem(self, indexes):
        '''Return a System instance in which only the given atom are retained.'''

        def reduce_array(old):
            if old is None:
                return None
            else:
                new = np.zeros((len(indexes),) + old.shape[1:], old.dtype)
                for inew, iold in enumerate(indexes):
                    new[inew] = old[iold]
                return new

        def reduce_scopes():
            if self.scopes is None:
                return None
            else:
                return [self.get_scope(i) for i in indexes]

        def reduce_ffatypes():
            if self.ffatypes is None:
                return None
            else:
                return [self.get_ffatype(i) for i in indexes]

        def reduce_bonds(old):
            translation = dict((iold, inew) for inew, iold in enumerate(indexes))
            new = []
            for old0, old1 in old:
                new0 = translation.get(old0)
                new1 = translation.get(old1)
                if not (new0 is None or new1 is None):
                    new.append([new0, new1])
            return new

        return System(
            numbers=reduce_array(self.numbers),
            pos=reduce_array(self.pos),
            scopes=reduce_scopes(),
            ffatypes=reduce_ffatypes(),
            bonds=reduce_bonds(self.bonds),
            rvecs=self.cell.rvecs,
            charges=reduce_array(self.charges),
            radii=reduce_array(self.radii),
            valence_charges=reduce_array(self.valence_charges),
            dipoles=reduce_array(self.dipoles),
            radii2=reduce_array(self.radii2),
            masses=reduce_array(self.masses),
        )

    def cut_bonds(self, indexes):
        '''Remove all bonds of a fragment with the remainder of the system;

           **Arguments:**

           indexes
                The atom indexes in the fragment
        '''
        new_bonds = []
        indexes = set(indexes)
        for i0, i1 in self.bonds:
            if not ((i0 in indexes) ^ (i1 in indexes)):
                new_bonds.append([i0, i1])
        self.bonds = np.array(new_bonds)

    def iter_matches(self, other, overlapping=True):
        """Yield all renumberings of atoms that map the given system on the current.

        Parameters
        ----------
        other : yaff.System
            Another system with the same number of atoms (and chemical formula), or less
            atoms.
        overlapping : bool
            When set to False, the returned matches are guaranteed to be mutually
            exclusive. The result may not be unique when partially overlapping matches
            would exist. Use with care.

        The graph distance is used to perform the mapping, so bonds must be defined in
        the current and the given system.
        """
        def make_graph_distance_matrix(system):
            """Return a bond graph distance matrix.

            Parameters
            ----------
            system : System
                Molecule (with bonds) for which the graph distances must be computed.

            The graph distance is used for comparison because it allows the pattern
            matching to make optimal choices of which pairs of atoms to compare next, i.e.
            both bonded or nearby the last matched pair.
            """
            from molmod.graphs import Graph
            return Graph(system.bonds, system.natom).distances

        def error_sq_fn(x, y):
            """Compare bonded versus not bonded, rather than the full graph distance.

            Parameters
            ----------
            x, y: int
                Graph distances from self and other, respectively.

            Graph distances are not completely transferable between self and other, i.e. a
            shorter path may exist between two atoms in the big system (self) that is not
            present in a fragment (other). Hence, only the absence or presence of a direct
            bond must be compared.
            """
            return (min(x - 1, 1) - min(y - 1, 1))**2

        with log.section('SYS'):
            log('Generating allowed indexes for renumbering.')
            # The allowed permutations is just based on the chemical elements, not the atom
            # types, which could also be useful.
            allowed = []
            if self.ffatypes is None or other.ffatypes is None:
                for number1 in other.numbers:
                    allowed.append((self.numbers == number1).nonzero()[0])
            else:
                # Only continue if other.ffatypes is a subset of self.ffatypes
                if not (set(self.ffatypes) >= set(other.ffatypes)):
                    return
                ffatype_ids0 = self.ffatype_ids
                ffatypes0 = list(self.ffatypes)
                order = np.array([ffatypes0.index(ffatype) for ffatype in other.ffatypes])
                ffatype_ids1 = order[other.ffatype_ids]
                for ffatype_id1 in ffatype_ids1:
                    allowed.append((ffatype_ids0 == ffatype_id1).nonzero()[0])
            log('Building distance matrix for self.')
            dm0 = make_graph_distance_matrix(self)
            log('Building distance matrix for other.')
            dm1 = make_graph_distance_matrix(other)
            # Yield the solutions
            log('Generating renumberings.')
            for match in iter_matches(dm0, dm1, allowed, 1e-3, error_sq_fn, overlapping):
                yield match

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
                'radii': self.radii,
                'valence_charges': self.valence_charges,
                'dipoles': self.dipoles,
                'radii2': self.radii2,
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
            sgrp.create_dataset('scopes', data=self.scopes.astype('S22'))
            sgrp.create_dataset('scope_ids', data=self.scope_ids)
        if self.ffatypes is not None:
            sgrp.create_dataset('ffatypes', data=self.ffatypes.astype('S22'))
            sgrp.create_dataset('ffatype_ids', data=self.ffatype_ids)
        if self.bonds is not None:
            sgrp.create_dataset('bonds', data=self.bonds)
        if self.cell.nvec > 0:
            sgrp.create_dataset('rvecs', data=self.cell.rvecs)
        if self.charges is not None:
            sgrp.create_dataset('charges', data=self.charges)
        if self.radii is not None:
            sgrp.create_dataset('radii', data=self.radii)
        if self.valence_charges is not None:
            sgrp.create_dataset('valence_charges', data=self.charges)
        if self.dipoles is not None:
            sgrp.create_dataset('dipoles', data=self.dipoles)
        if self.radii2 is not None:
            sgrp.create_dataset('radii2', data=self.radii2)
        if self.masses is not None:
            sgrp.create_dataset('masses', data=self.masses)
