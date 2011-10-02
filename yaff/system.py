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

from yaff.topology import Topology
from yaff.ext import Cell
from yaff.log import log


__all__ = ['System']


class System(object):
    def __init__(self, numbers, pos, ffatypes, bonds=None, rvecs=None):
        '''
           **Arguments:**

           numbers
                A numpy array with atomic numbers

           pos
                A numpy array (N,3) with atomic coordinates in bohr.

           ffatypes
                A list of labels of the force field atom types.

           **Optional arguments:**

           bonds
                a numpy array (B,2) with atom indexes (counting starts from
                zero) to define the chemical bonds.

           rvecs
                An array whose rows are the unit cell vectors. At most three
                rows are allowed, each containg three Cartesian coordinates.
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
        if bonds is None:
            self.topology = None
        else:
            self.topology = Topology(bonds, self.natom)
        self.cell = Cell(rvecs)

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
                kwargs['bonds'] = psf.bonds
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
            'bonds': self.topology.bonds,
            'rvecs': self.cell.rvecs,
        })
        if log.do_high:
            log('SYS', 'Wrote system to %s.' % fn_chk)
