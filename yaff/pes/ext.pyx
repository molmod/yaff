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
#cython: embedsignature=True
'''Low-level C routines

   This extension module is used by various modules of the ``yaff.pes``
   package.
'''


from __future__ import division


import numpy as np
cimport numpy as np
cimport cell
cimport nlist
cimport pair_pot
cimport ewald
cimport dlist
cimport iclist
cimport vlist
cimport truncation
cimport grid

from yaff.log import log


__all__ = [
    'Cell', 'nlist_status_init', 'nlist_build', 'nlist_status_finish',
    'nlist_recompute', 'nlist_inc_r', 'Hammer', 'Switch3', 'PairPot',
    'PairPotLJ', 'PairPotMM3', 'PairPotGrimme', 'PairPotExpRep',
    'PairPotQMDFFRep', 'PairPotLJCross', 'PairPotDampDisp',
    'PairPotDisp68BJDamp', 'PairPotEI', 'PairPotEIDip',
    'PairPotEiSlater1s1sCorr', 'PairPotEiSlater1sp1spCorr',
    'PairPotOlpSlater1s1s','PairPotChargeTransferSlater1s1s',
    'compute_ewald_reci', 'compute_ewald_reci_dd',  'compute_ewald_corr_dd',
    'compute_ewald_corr', 'dlist_forward', 'dlist_back', 'iclist_forward',
    'iclist_back', 'vlist_forward', 'vlist_back', 'compute_grid3d',
]


#
# Cell
#

cdef class Cell:
    '''Representation of periodic boundary conditions.

       0, 1, 2 and 3 dimensional systems are supported. The cell vectors need
       not to be orthogonal.
    '''
    cdef cell.cell_type* _c_cell

    def __cinit__(self, *args, **kwargs):
        self._c_cell = cell.cell_new()
        if self._c_cell is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_cell is not NULL:
            cell.cell_free(self._c_cell)

    def __init__(self, np.ndarray[double, ndim=2] rvecs):
        '''
           **Arguments:**

           rvecs
                A numpy array with at most three cell vectors, layed out as
                rows in a rank-2 matrix. For non-periodic systems, this array
                must have shape (0,3).
        '''
        self.update_rvecs(rvecs)

    def update_rvecs(self, np.ndarray[double, ndim=2] rvecs):
        '''Change the cell vectors and recompute the reciprocal cell vectors.

           rvecs
                A numpy array with at most three cell vectors, layed out as
                rows in a rank-2 matrix. For non-periodic systems, this array
                must have shape (0,3).
        '''
        cdef np.ndarray[double, ndim=2] mod_rvecs
        cdef np.ndarray[double, ndim=2] gvecs
        cdef int nvec
        if rvecs is None or rvecs.size == 0:
            mod_rvecs = np.identity(3, float)
            gvecs = mod_rvecs
            nvec = 0
        else:
            if not rvecs.ndim==2 or not rvecs.flags['C_CONTIGUOUS'] or rvecs.shape[0] > 3 or rvecs.shape[1] != 3:
                raise TypeError('rvecs must be a C contiguous array with three columns and at most three rows.')
            nvec = len(rvecs)
            Up, Sp, Vt = np.linalg.svd(rvecs, full_matrices=True)
            S = np.ones(3, float)
            S[:nvec] = Sp
            U = np.identity(3, float)
            U[:nvec,:nvec] = Up
            mod_rvecs = np.dot(U*S, Vt)
            mod_rvecs[:nvec] = rvecs
            gvecs = np.dot(U/S, Vt)
        cell.cell_update(self._c_cell, <double*>mod_rvecs.data, <double*>gvecs.data, nvec)

    def _get_nvec(self):
        '''The number of cell vectors'''
        return cell.cell_get_nvec(self._c_cell)

    nvec = property(_get_nvec)

    def _get_volume(self):
        '''The generalize volume of the unit cell (length, area or volume)'''
        return cell.cell_get_volume(self._c_cell)

    volume = property(_get_volume)

    def _get_rvecs(self, full=False):
        '''The real-space cell vectors, layed out as rows.'''
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_rvecs(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rvecs = property(_get_rvecs)

    def _get_gvecs(self, full=False):
        '''The reciporcal-space cell vectors, layed out as rows.'''
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_gvecs(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gvecs = property(_get_gvecs)

    def _get_rspacings(self, full=False):
        '''The (orthogonal) spacing between opposite sides of the real-space unit cell.'''
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_rspacings(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rspacings = property(_get_rspacings)

    def _get_gspacings(self, full=False):
        '''The (orthogonal) spacing between opposite sides of the reciprocal-space unit cell.'''
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_gspacings(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gspacings = property(_get_gspacings)

    def _get_parameters(self):
        '''The cell parameters (lengths and angles)'''
        rvecs = self._get_rvecs()
        tmp = np.dot(rvecs, rvecs.T)
        lengths = np.sqrt(np.diag(tmp))
        tmp /= lengths
        tmp /= lengths.reshape((-1,1))
        if len(rvecs) < 2:
            cosines = np.array([])
        elif len(rvecs) == 2:
            cosines = np.array([tmp[0,1]])
        else:
            cosines = np.array([tmp[1,2], tmp[2,0], tmp[0,1]])
        angles = np.arccos(np.clip(cosines, -1, 1))
        return lengths, angles

    parameters = property(_get_parameters)

    def mic(self, np.ndarray[double, ndim=1] delta):
        """Apply the minimum image convention to delta in-place"""
        assert delta.size == 3
        cell.cell_mic(<double*> delta.data, self._c_cell)

    def to_center(self, np.ndarray[double, ndim=1] pos):
        '''Return the corresponding position in the central cell'''
        assert pos.size == 3
        cdef np.ndarray[long, ndim=1] result
        result = np.zeros(self.nvec, int)
        cell.cell_to_center(<double*> pos.data, self._c_cell, <long*> result.data)
        return result

    def add_vec(self, np.ndarray[double, ndim=1] delta, np.ndarray[long, ndim=1] r):
        """Add a linear combination of cell vectors, ``r``, to ``delta`` in-place
        """
        assert delta.size == 3
        assert r.size == self.nvec
        cell.cell_add_vec(<double*> delta.data, self._c_cell, <long*> r.data)

    def compute_distances(self, np.ndarray[double, ndim=1] output,
                          np.ndarray[double, ndim=2] pos0,
                          np.ndarray[double, ndim=2] pos1=None,
                          np.ndarray[long, ndim=2] pairs=None,
                          bint do_include=False,
                          long nimage=0):
        """Computes all distances between the given coordinates

           **Arguments:**

           output
                An numpy vector of the proper length that will be used to store
                all the distances.

           pos0
                An array with Cartesian coordinates

           **Optional arguments:**

           pos1
                A second array with Cartesian coordinates

           pairs
                A sorted array of atom pairs. When do_include==False, this list
                will be excluded from the computation. When do_include==True,
                only these pairs are considered when computing distances.

                The indexes in this array refer to rows of pos0 or pos1. If pos1
                is not given, both columns refer to rows of pos0. If pos1 is
                given, the first column refers to rows of pos0 and the second
                column refers to rows of pos1. The rows in the pairst array
                should be sorted lexicographically, first along the first
                column, then along the second column.

           do_include
                True or False, controls how the pairs list is interpreted. When
                set to True, nimage must be zero and the pairs attribute must be
                a non-empty array.

           nimage
                The number of cell images to consider in the computation of the
                pair distances. By default, this is zero, meaning that only the
                minimum image convention is used.

           This routine can operate in two different ways, depending on the
           presence/absence of the argument ``pos1``. If not given, all
           distances between points in ``pos0`` are computed and the length of
           the output array is ``len(pos0)*(len(pos0)-1)/2``. If ``pos1`` is
           given, all distances are computed between a point in ``pos0`` and a
           point in ``pos1`` and the length of the output array is
           ``len(pos0)*len(pos1)``.

           In both cases, some pairs of atoms may be
           excluded from the output with the ``exclude`` argument. In typical
           cases, this list of excluded pairs is relatively short. In case,
           the exclude argument is present the number of computed distances
           is less than explained above, but it is recommended to still use
           those sizes in case some pairs in the excluded list are not
           applicable.
        """
        cdef long* pairs_pointer

        assert pos0.shape[1] == 3
        assert pos0.flags['C_CONTIGUOUS']
        assert nimage >= 0
        natom0 = pos0.shape[0]

        if pairs is not None:
            assert pairs.shape[1] == 2
            assert pairs.flags['C_CONTIGUOUS']
            pairs_pointer = <long*> pairs.data
            npair = pairs.shape[0]
        else:
            pairs_pointer = NULL
            npair = 0

        if nimage > 0:
            if self.nvec == 0:
                raise ValueError('Can only include distances to periodic images for periodic systems.')
            factor = (1+2*nimage)**self.nvec
        else:
            factor = 1

        if do_include:
            if nimage != 0:
                raise ValueError('When do_include==True, nimage must be zero.')
            if npair == 0:
                raise ValueError('No pairs given and do_include==True.')

        if pos1 is None:
            if do_include:
                npair == output.shape[0]
            else:
                assert factor*(natom0*(natom0-1))//2 - npair == output.shape[0]
            if cell.is_invalid_exclude(pairs_pointer, natom0, natom0, npair, True):
                raise ValueError('The pairs array must countain indices within proper bounds and must be lexicographically sorted.')
            cell.cell_compute_distances1(self._c_cell, <double*> pos0.data,
                                         <double*> output.data, natom0,
                                         <long*> pairs_pointer, npair, do_include, nimage)
        else:
            assert pos1.shape[1] == 3
            assert pos1.flags['C_CONTIGUOUS']
            natom1 = pos1.shape[0]

            if do_include:
                npair == output.shape[0]
            else:
                assert factor*natom0*natom1 - npair == output.shape[0]
            if cell.is_invalid_exclude(pairs_pointer, natom0, natom1, npair, False):
                raise ValueError('The pairs array must countain indices within proper bounds and must be lexicographically sorted.')
            cell.cell_compute_distances2(self._c_cell, <double*> pos0.data,
                                         <double*> pos1.data,
                                         <double*> output.data, natom0, natom1,
                                         <long*> pairs_pointer, npair, do_include, nimage)


#
# Neighbor lists
#


def nlist_status_init(rmax):
    '''nlist_status_init(rmax)

       Creates a new ``nlists_status`` array

       The array consists of seven integer elements with the following meaning:

       * ``r0``: relative image index along a direction
       * ``r1``: relative image index along b direction
       * ``r2``: relative image index along c direction
       * ``a``: atom index of first atom in pair
       * ``b``: atom index of second atom in pair
       * ``sign``: +1 or -1, to swap the relative vector such that a > b
       * ``nrow``: number of rows consumed
    '''
    result = np.array([0, 0, 0, 0, 0, 1, 0], int)
    return result


def nlist_build(np.ndarray[double, ndim=2] pos, double rcut,
                np.ndarray[long, ndim=1] rmax,
                Cell unitcell, np.ndarray[long, ndim=1] status,
                np.ndarray[nlist.neigh_row_type, ndim=1] neighs):
    '''Scan the system for all pairs that have a distance smaller than rcut until the neighs array is filled or all pairs are considered

       **Arguments:**

       pos
            The numpy array with the atomic positions, shape (natom, 3)

       rcut
            The cutoff radius

       rmax
            The number of periodic images to visit along each cell vector, shape
            (nrvec,)

       unitcell
            An instance of the UnitCell class, describing the periodic boundary
            conditions.

       status
            The status array, either obtained from ``nlist_status_init``, or
            as it was modified by the last call to this function

       neighs
            The neighbor list array. One element is of the datatype
            nlist.neigh_row_type.

       **Returns:**

       ``True`` if the neighbor list is complete. ``False`` otherwise
    '''
    assert pos.shape[1] == 3
    assert pos.flags['C_CONTIGUOUS']
    assert rcut > 0
    assert rmax.shape[0] <= 3
    assert rmax.flags['C_CONTIGUOUS']
    assert status.shape[0] == 7
    assert status.flags['C_CONTIGUOUS']
    assert neighs.flags['C_CONTIGUOUS']
    assert rmax.shape[0] == unitcell.nvec
    return nlist.nlist_build_low(
        <double*>pos.data, rcut, <long*>rmax.data,
        unitcell._c_cell, <long*>status.data,
        <nlist.neigh_row_type*>neighs.data, len(pos), len(neighs)
    )


def nlist_status_finish(status):
    '''status
            The status array, either obtained from ``nlist_status_init``, or
            as it was modified by the last call to this function

       Returns the number of rows generated by the neighbor list algorithm
    '''
    return status[-1]


def nlist_recompute(np.ndarray[double, ndim=2] pos,
                    np.ndarray[double, ndim=2] pos_old,
                    Cell unitcell,
                    np.ndarray[nlist.neigh_row_type, ndim=1] neighs):
    '''Recompute all relative vectors and distances in the neighbor list.

       **Arguments:**

       pos
            The numpy array with the atomic positions, shape (natom, 3)

       pos_old
            The positions used during the last neighbor list rebuild. These
            are used to make sure that there are no sudden jumps in the
            relative vectors due to the minimum image convention.

       unitcell
            An instance of the UnitCell class, describing the periodic boundary
            conditions.

       neighs
            The neighbor list array. One element is of the datatype
            nlist.neigh_row_type.
    '''
    assert pos.shape[1] == 3
    assert pos.flags['C_CONTIGUOUS']
    assert pos_old.shape[1] == 3
    assert pos_old.flags['C_CONTIGUOUS']
    assert pos.shape[0] == pos_old.shape[0]
    assert neighs.flags['C_CONTIGUOUS']
    nlist.nlist_recompute_low(
        <double*>pos.data, <double*>pos_old.data, unitcell._c_cell,
        <nlist.neigh_row_type*>neighs.data, len(neighs)
    )


def nlist_inc_r(Cell unitcell, np.ndarray[long, ndim=1] r, np.ndarray[long, ndim=1] rmax):
    '''Increment the vector ``r`` to the location of the `next` periodic image.

       **Arguments:**

       unitcell
            An instance of the UnitCell class, describing the periodic boundary
            conditions.

       r
            An array of integers describing the current image. This will be
            incremented in place.

       rmax
            An array of integers specifying the range of periodic images that
            must be visited along each cell vector.

       **Returns:**

       True if the counter ``r`` was at the last image and is therefore reset
       to the first image. False in all other cases.

       **Description:**

       This Python wrapper is only present for debugging purposes. Note that
       this routine visits only half of the periodic images because the other
       half contains exactly the same relative vectors.
    '''
    return nlist.nlist_inc_r(unitcell._c_cell, <long*>r.data, <long*>rmax.data)


#
# Pair potential truncation schemes
#


cdef class Truncation:
    '''Base class for truncation schemes of pairwise interactions'''
    cdef truncation.trunc_scheme_type* _c_trunc_scheme

    def __dealloc__(self):
        if self._c_trunc_scheme is not NULL:
            truncation.trunc_scheme_free(self._c_trunc_scheme)

    def trunc_fn(self, double d, double rcut):
        '''trunc_fn(d, rcut)

           Return the truncation function and its derivative.

           **Arguments:**

           d
                The distance at which the truncation function must be evaluated.

           rcut
                The cutoff radius.
        '''
        cdef double hg
        hg = 0.0
        h = truncation.trunc_scheme_fn(self._c_trunc_scheme, d, rcut, &hg)
        return h, hg


cdef class Hammer(Truncation):
    r'''An old-fashioned and poor truncation scheme.

       **Arguments:**

       tau
            The tau parameter in the mathematical expression below.

       Don't use this truncation scheme. Only present for historical reasons.
       The mathematical form is as follows:

       .. math:: t_\text{hammer}(d) = \left\lbrace \begin{array}{ll}
                     \exp\left(\frac{\tau}{d-r_\text{rcut}}\right) & \text{if } d < r_\text{cut} \\
                     0 & \text{if } d >= r_\text{cut}
                 \end{array} \right.
    '''
    def __cinit__(self, double tau):
        self._c_trunc_scheme = truncation.hammer_new(tau)
        if self._c_trunc_scheme is NULL:
            raise MemoryError

    def _get_tau(self):
        '''The Tau parameter of the truncation function'''
        return truncation.hammer_get_tau(self._c_trunc_scheme)

    tau = property(_get_tau)

    def get_log(self):
        '''get_log()

           Return a string suitable for the screen logger
        '''
        return 'hammer %s' % log.length(self.tau)


cdef class Switch3(Truncation):
    r'''A simple and good truncation scheme.

       **Arguments:**

       width
            The width parameter, :math:`w`, in the mathematical expression below.

       This is the recommended truncation scheme in Yaff. It has the following
       mathematical form:

       .. math:: t_\text{swithc3}(d) = \left\lbrace \begin{array}{ll}
                     1 & \text{if } d < r_\text{cut} - w \\
                     3x^2 - 2x^3 & \text{if } d < r_\text{cut} \text{ with } x=\frac{r_\text{cut} - d}{w} \\
                     0 & \text{if } d >= r_\text{cut}
                 \end{array} \right.
    '''
    def __cinit__(self, double width):
        self._c_trunc_scheme = truncation.switch3_new(width)
        if self._c_trunc_scheme is NULL:
            raise MemoryError

    def _get_width(self):
        '''The width parameter of the truncation scheme'''
        return truncation.switch3_get_width(self._c_trunc_scheme)

    width = property(_get_width)

    def get_log(self):
        '''get_log()

           Return a string suitable for the screen logger
        '''
        return 'switch3 %s' % log.length(self.width)


#
# Pair potentials
#


cdef class PairPot:
    '''Base class for the pair potentials'''
    cdef pair_pot.pair_pot_type* _c_pair_pot
    cdef Truncation tr

    def __cinit__(self, *args, **kwargs):
        self._c_pair_pot = pair_pot.pair_pot_new()
        if self._c_pair_pot is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if pair_pot.pair_pot_ready(self._c_pair_pot):
            pair_pot.pair_data_free(self._c_pair_pot)
        if self._c_pair_pot is not NULL:
            pair_pot.pair_pot_free(self._c_pair_pot)

    def _get_rcut(self):
        '''The cutoff parameter'''
        return pair_pot.pair_pot_get_rcut(self._c_pair_pot)

    rcut = property(_get_rcut)

    cdef set_truncation(self, Truncation tr):
        '''Set the truncation scheme'''
        self.tr = tr
        if tr is None:
            pair_pot.pair_pot_set_trunc_scheme(self._c_pair_pot, NULL)
        else:
            pair_pot.pair_pot_set_trunc_scheme(self._c_pair_pot, tr._c_trunc_scheme)

    def get_truncation(self):
        '''Returns the current truncation scheme'''
        return self.tr

    def compute(self, np.ndarray[nlist.neigh_row_type, ndim=1] neighs,
                np.ndarray[pair_pot.scaling_row_type, ndim=1] stab,
                np.ndarray[double, ndim=2] gpos,
                np.ndarray[double, ndim=2] vtens, long nneigh):
        '''Compute the pairwise interactions

           **Arguments:**

           neighs
                The neighbor list array. One element is of the datatype
                nlist.neigh_row_type.

           stab
                The array with short-range scalings. Each element is of the
                datatype pair_pot.scaling_row_type

           gpos
                The output array for the derivative of the energy towards the
                atomic positions. If None, these derivatives are not computed.

           vtens
                The output array for the virial tensor. If none, it is not
                computed.

           nneigh
                The number of records to consider in the neighbor list.

           **Returns:** the energy.
        '''
        cdef double *my_gpos
        cdef double *my_vtens

        assert pair_pot.pair_pot_ready(self._c_pair_pot)
        assert neighs.flags['C_CONTIGUOUS']
        assert stab.flags['C_CONTIGUOUS']

        if gpos is None:
            my_gpos = NULL
        else:
            assert gpos.flags['C_CONTIGUOUS']
            assert gpos.shape[1] == 3
            my_gpos = <double*>gpos.data

        if vtens is None:
            my_vtens = NULL
        else:
            assert vtens.flags['C_CONTIGUOUS']
            assert vtens.shape[0] == 3
            assert vtens.shape[1] == 3
            my_vtens = <double*>vtens.data

        return pair_pot.pair_pot_compute(
            <nlist.neigh_row_type*>neighs.data, nneigh,
            <pair_pot.scaling_row_type*>stab.data, len(stab),
            self._c_pair_pot, my_gpos, my_vtens
        )


cdef class PairPotLJ(PairPot):
    r'''Lennard-Jones pair potential:

       **Energy:**

       .. math:: E_\text{LJ} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} 4 s_{ij} \epsilon_{ij} \left[
                 \left(\frac{\sigma_{ij}}{d_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{d_{ij}}\right)^6
                 \right]

       with

       .. math:: \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}

       .. math:: \sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}

       .. math:: s_{ij} = \text{the short-range scaling factor}


       **Arguments:**

       sigmas
            An array with sigma parameters, one for each atom, shape (natom,)

       epsilons
            An array with epsilon parameters, one for each atom, shape (natom,)

       rcut
            The cutoff radius

       **Optional arguments:**

       tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied
    '''
    cdef np.ndarray _c_sigmas
    cdef np.ndarray _c_epsilons
    name = 'lj'

    def __cinit__(self, np.ndarray[double, ndim=1] sigmas,
                  np.ndarray[double, ndim=1] epsilons, double rcut,
                  Truncation tr=None):
        assert sigmas.flags['C_CONTIGUOUS']
        assert epsilons.flags['C_CONTIGUOUS']
        assert sigmas.shape[0] == epsilons.shape[0]
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_lj_init(self._c_pair_pot, <double*>sigmas.data, <double*>epsilons.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_sigmas = sigmas
        self._c_epsilons = epsilons

    def log(self):
        '''Write some suitable post-initialization screen log'''
        if log.do_high:
            log.hline()
            log('   Atom      Sigma    Epsilon')
            log.hline()
            for i in range(self._c_sigmas.shape[0]):
                log('%7i %s %s' % (i, log.length(self._c_sigmas[i]), log.energy(self._c_epsilons[i])))

    def _get_sigmas(self):
        '''The array with sigma parameters'''
        return self._c_sigmas.view()

    sigmas = property(_get_sigmas)

    def _get_epsilons(self):
        '''The array with epsilon parameters'''
        return self._c_epsilons.view()

    epsilons = property(_get_epsilons)


cdef class PairPotMM3(PairPot):
    r'''The MM3 version of the Lennard-Jones pair potential

       **Energy:**

       .. math:: E_\text{MM3} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} \epsilon_{ij} \left[
                 1.84\times10^{5} \exp\left(\frac{\sigma_{ij}}{d_{ij}}\right) - 2.25\left(\frac{\sigma_{ij}}{d_{ij}}\right)^6
                 \right]

       with

       .. math:: \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}

       .. math:: \sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}

       .. math:: s_{ij} = \text{the short-range scaling factor}

       **Arguments:**

       sigmas
            An array with sigma parameters, one for each atom, shape (natom,)

       epsilons
            An array with epsilon parameters, one for each atom, shape (natom,)

       onlypaulis
            An array integers. When non-zero for both atoms in a pair, only the
            repulsive wall is computed.

       rcut
            The cutoff radius

       **Optional arguments:**

       tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied
    '''
    cdef np.ndarray _c_sigmas
    cdef np.ndarray _c_epsilons
    cdef np.ndarray _c_onlypaulis
    name = 'mm3'

    def __cinit__(self, np.ndarray[double, ndim=1] sigmas,
                  np.ndarray[double, ndim=1] epsilons,
                  np.ndarray[int, ndim=1] onlypaulis, double rcut,
                  Truncation tr=None):
        assert sigmas.flags['C_CONTIGUOUS']
        assert epsilons.flags['C_CONTIGUOUS']
        assert onlypaulis.flags['C_CONTIGUOUS']
        assert sigmas.shape[0] == epsilons.shape[0]
        assert sigmas.shape[0] == onlypaulis.shape[0]
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_mm3_init(self._c_pair_pot, <double*>sigmas.data, <double*>epsilons.data, <int*>onlypaulis.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_sigmas = sigmas
        self._c_epsilons = epsilons
        self._c_onlypaulis = onlypaulis

    def log(self):
        '''Write some suitable post-initialization screen log'''
        if log.do_high:
            log.hline()
            log('   Atom      Sigma    Epsilon    OnlyPauli')
            log.hline()
            for i in range(self._c_sigmas.shape[0]):
                log('%7i %s %s            %i' % (i, log.length(self._c_sigmas[i]), log.energy(self._c_epsilons[i]), self._c_onlypaulis[i]))

    def _get_sigmas(self):
        '''The array with sigma parameters'''
        return self._c_sigmas.view()

    sigmas = property(_get_sigmas)

    def _get_epsilons(self):
        '''The array with epsilon parameters'''
        return self._c_epsilons.view()

    epsilons = property(_get_epsilons)

    def _get_onlypaulis(self):
        '''The array with the only-Pauli flag'''
        return self._c_onlypaulis.view()

    onlypaulis = property(_get_onlypaulis)


cdef class PairPotGrimme(PairPot):
    cdef np.ndarray _c_r0
    cdef np.ndarray _c_c6
    name = 'grimme'

    def __cinit__(self, np.ndarray[double, ndim=1] r0,
                  np.ndarray[double, ndim=1] c6, double rcut,
                  Truncation tr=None):
        assert r0.flags['C_CONTIGUOUS']
        assert c6.flags['C_CONTIGUOUS']
        assert r0.shape[0] == c6.shape[0]
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_grimme_init(self._c_pair_pot, <double*>r0.data, <double*>c6.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_r0 = r0
        self._c_c6 = c6

    def log(self):
        if log.do_high:
            log.hline()
            log('   Atom         r0         c6')
            log.hline()
            for i in range(self._c_r0.shape[0]):
                log('%7i %s %s' % (i, log.length(self._c_r0[i]), log.c6(self._c_c6[i])))

    def _get_r0(self):
        return self._c_r0.view()

    r0 = property(_get_r0)

    def _get_c6(self):
        return self._c_c6.view()

    c6 = property(_get_c6)


cdef class PairPotExpRep(PairPot):
    r'''Exponential repulsion

        .. math:: E_\text{EXPREP} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} A_{ij} \exp(-B_{ij} d_{ij})

        The pair parameters can be provided explicitly, or can be derived from atomic
        parameters using two possible mixing rules for each parameter:

        * ``GEOMETRIC`` mixing for :math:`A_{ij}`: :math:`A_{ij} = \sqrt{A_i A_j}`

        * ``GEOMETRIC_COR`` mixing for :math:`A_{ij}`: :math:`\ln A_{ij} = (\ln A_i + \ln A_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

        * ``ARITHMETIC`` mixing for :math:`B_{ij}`: :math:`B_{ij} = \frac{B_i + B_j}{2}`

        * ``ARITHMETIC_COR`` mixing for :math:`B_{ij}`: :math:`B_{ij} = (B_i + B_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

        **Arguments:**

        ffatype_ids
            An array with atom type IDs for each atom. The IDs are integer
            indexes for the atom types that start counting from zero. shape =
            (natom,).

        amp_cross
            A 2D array of amplitude cross parameters (:math:`A_{ij}`)

        b_cross
            A 2D array of decay cross parameters (:math:`B_{ij}`)

        rcut
             The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        amps
            A 1D array of amplitude diagonal parameters (:math:`A_{i}`)

        amp_mix
            An integer ID that determines the mixing rule for the a parameter.
            0=GEOMETRIC, 1=GEOMETRIC_COR.

        amp_mix_coeff
            The parameter :math:`x` for the corrected geometric mixing rule.

        bs
            A 1D array of decay diagonal parameters (:math:`B_{i}`)

        b_mix
            An integer ID that determines the mixing rule for the B parameter.
            0=ARITHMETIC, 1=ARITHMETIC_COR.

        b_mix_coeff
            The parameter :math:`x` for the corrected arithmetic mixing rule.

        The mixing rules are only in effect when the optional diagonal
        parameters are given. Only when the cross parameters are zero (in the
        arguments ``amp_cross`` and ``b_cross``), these numbers would be
        overwritten by the mixing rules.
    '''
    cdef long _c_nffatype
    cdef np.ndarray _c_ffatype_ids
    cdef np.ndarray _c_amp_cross
    cdef np.ndarray _c_b_cross
    name = 'exprep'

    def __cinit__(self, np.ndarray[long, ndim=1] ffatype_ids not None,
                  np.ndarray[double, ndim=2] amp_cross not None,
                  np.ndarray[double, ndim=2] b_cross not None,
                  double rcut, Truncation tr=None,
                  np.ndarray[double, ndim=1] amps=None, long amp_mix=-1, double amp_mix_coeff=-1,
                  np.ndarray[double, ndim=1] bs=None, long b_mix=-1, double b_mix_coeff=-1):
        assert ffatype_ids.flags['C_CONTIGUOUS']
        assert amp_cross.flags['C_CONTIGUOUS']
        assert b_cross.flags['C_CONTIGUOUS']
        nffatype = amp_cross.shape[0]
        assert amp_cross.shape[1] == nffatype
        assert b_cross.shape[0] == nffatype
        assert b_cross.shape[1] == nffatype
        assert ffatype_ids.min() >= 0
        assert ffatype_ids.max() < nffatype
        if amps is not None:
            assert amps.shape[0] == nffatype
            assert amp_mix == 0 or amp_mix == 1
            assert amp_mix_coeff >= 0 and amp_mix_coeff <= 1
            self._init_amp_cross(nffatype, amp_cross, amps, amp_mix, amp_mix_coeff)
        assert (amp_cross == amp_cross.T).all()
        if bs is not None:
            assert bs.shape[0] == nffatype
            assert b_mix == 0 or b_mix == 1
            assert b_mix_coeff >= 0 and b_mix_coeff <= 1
            self._init_b_cross(nffatype, b_cross, bs, b_mix, b_mix_coeff, amps)
        assert (b_cross == b_cross.T).all()
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_exprep_init(
            self._c_pair_pot, nffatype, <long*> ffatype_ids.data,
            <double*> amp_cross.data, <double*> b_cross.data
        )
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_nffatype = nffatype
        self._c_amp_cross = amp_cross
        self._c_b_cross = b_cross

    def _init_amp_cross(self, nffatype, amp_cross, amps, amp_mix, amp_mix_coeff):
        for i0 in range(nffatype):
            for i1 in range(i0+1):
                if amp_cross[i0, i1] == 0.0:
                    if amp_mix == 0:
                        amp_cross[i0, i1] = np.sqrt(amps[i0]*amps[i1])
                    elif amps[i0] == 0.0 or amps[i1] == 0.0:
                        amp = 0.0
                    else:
                        amp = (np.log(amps[i0])+np.log(amps[i1]))/2;
                        amp *= 1 - amp_mix_coeff*abs(np.log(amps[i0]/amps[i1]));
                        amp_cross[i0, i1] = np.exp(amp)
                    amp_cross[i1, i0] = amp_cross[i0, i1]

    def _init_b_cross(self, nffatype, b_cross, bs, b_mix, b_mix_coeff, amps):
        for i0 in range(nffatype):
            for i1 in range(i0+1):
                if b_cross[i0, i1] == 0.0:
                    if b_mix == 0:
                        b_cross[i0, i1] = (bs[i0] + bs[i1])/2
                    elif amps[i0] == 0.0 or amps[i1] == 0.0:
                        b = 0.0
                    else:
                        b = (bs[i0] + bs[i1])/2;
                        b *= 1 - b_mix_coeff*abs(np.log(amps[i0]/amps[i1]));
                        b_cross[i0, i1] = b
                    b_cross[i1, i0] = b_cross[i0, i1]

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1          A          B')
            log.hline()
            for i0 in range(self._c_nffatype):
                for i1 in range(i0+1):
                    log('%11i %11i %s %s' % (i0, i1, log.energy(self._c_amp_cross[i0, i1]), log.invlength(self._c_b_cross[i0,i1])))

    def _get_amp_cross(self):
        '''The amplitude cross parameters'''
        return self._c_amp_cross.view()

    amp_cross = property(_get_amp_cross)

    def _get_b_cross(self):
        '''The decay cross parameters'''
        return self._c_b_cross.view()

    b_cross = property(_get_b_cross)


cdef class PairPotQMDFFRep(PairPot):
    r'''Exponential repulsion from QMDFF force field of Grimme

        .. math:: E_\text{EXPREP} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} \frac{A_{ij}}{d_{ij}} \exp(-B_{ij} d_{ij})

        The pair parameters can be provided explicitly, or can be derived from atomic
        parameters using two possible mixing rules for each parameter:

        * ``GEOMETRIC`` mixing for :math:`A_{ij}`: :math:`A_{ij} = \sqrt{A_i A_j}`

        * ``GEOMETRIC_COR`` mixing for :math:`A_{ij}`: :math:`\ln A_{ij} = (\ln A_i + \ln A_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

        * ``ARITHMETIC`` mixing for :math:`B_{ij}`: :math:`B_{ij} = \frac{B_i + B_j}{2}`

        * ``ARITHMETIC_COR`` mixing for :math:`B_{ij}`: :math:`B_{ij} = (B_i + B_j)\frac{1-x\vert\ln(A_i/A_j)\vert}{2}` where :math:`x` is a configurable parameter

        **Arguments:**

        ffatype_ids
            An array with atom type IDs for each atom. The IDs are integer
            indexes for the atom types that start counting from zero. shape =
            (natom,).

        amp_cross
            A 2D array of amplitude cross parameters (:math:`A_{ij}`)

        b_cross
            A 2D array of decay cross parameters (:math:`B_{ij}`)

        rcut
             The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        amps
            A 1D array of amplitude diagonal parameters (:math:`A_{i}`)

        amp_mix
            An integer ID that determines the mixing rule for the a parameter.
            0=GEOMETRIC, 1=GEOMETRIC_COR.

        amp_mix_coeff
            The parameter :math:`x` for the corrected geometric mixing rule.

        bs
            A 1D array of decay diagonal parameters (:math:`B_{i}`)

        b_mix
            An integer ID that determines the mixing rule for the B parameter.
            0=ARITHMETIC, 1=ARITHMETIC_COR.

        b_mix_coeff
            The parameter :math:`x` for the corrected arithmetic mixing rule.

        The mixing rules are only in effect when the optional diagonal
        parameters are given. Only when the cross parameters are zero (in the
        arguments ``amp_cross`` and ``b_cross``), these numbers would be
        overwritten by the mixing rules.
    '''
    cdef long _c_nffatype
    cdef np.ndarray _c_ffatype_ids
    cdef np.ndarray _c_amp_cross
    cdef np.ndarray _c_b_cross
    name = 'qmdffrep'

    def __cinit__(self, np.ndarray[long, ndim=1] ffatype_ids not None,
                  np.ndarray[double, ndim=2] amp_cross not None,
                  np.ndarray[double, ndim=2] b_cross not None,
                  double rcut, Truncation tr=None,
                  np.ndarray[double, ndim=1] amps=None, long amp_mix=-1, double amp_mix_coeff=-1,
                  np.ndarray[double, ndim=1] bs=None, long b_mix=-1, double b_mix_coeff=-1):
        assert ffatype_ids.flags['C_CONTIGUOUS']
        assert amp_cross.flags['C_CONTIGUOUS']
        assert b_cross.flags['C_CONTIGUOUS']
        nffatype = amp_cross.shape[0]
        assert amp_cross.shape[1] == nffatype
        assert b_cross.shape[0] == nffatype
        assert b_cross.shape[1] == nffatype
        assert ffatype_ids.min() >= 0
        assert ffatype_ids.max() < nffatype
        if amps is not None:
            assert amps.shape[0] == nffatype
            assert amp_mix == 0 or amp_mix == 1
            assert amp_mix_coeff >= 0 and amp_mix_coeff <= 1
            self._init_amp_cross(nffatype, amp_cross, amps, amp_mix, amp_mix_coeff)
        assert (amp_cross == amp_cross.T).all()
        if bs is not None:
            assert bs.shape[0] == nffatype
            assert b_mix == 0 or b_mix == 1
            assert b_mix_coeff >= 0 and b_mix_coeff <= 1
            self._init_b_cross(nffatype, b_cross, bs, b_mix, b_mix_coeff, amps)
        assert (b_cross == b_cross.T).all()
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_qmdffrep_init(
            self._c_pair_pot, nffatype, <long*> ffatype_ids.data,
            <double*> amp_cross.data, <double*> b_cross.data
        )
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_nffatype = nffatype
        self._c_amp_cross = amp_cross
        self._c_b_cross = b_cross

    def _init_amp_cross(self, nffatype, amp_cross, amps, amp_mix, amp_mix_coeff):
        for i0 in range(nffatype):
            for i1 in range(i0+1):
                if amp_cross[i0, i1] == 0.0:
                    if amp_mix == 0:
                        amp_cross[i0, i1] = np.sqrt(amps[i0]*amps[i1])
                    elif amps[i0] == 0.0 or amps[i1] == 0.0:
                        amp = 0.0
                    else:
                        amp = (np.log(amps[i0])+np.log(amps[i1]))/2;
                        amp *= 1 - amp_mix_coeff*abs(np.log(amps[i0]/amps[i1]));
                        amp_cross[i0, i1] = np.exp(amp)
                    amp_cross[i1, i0] = amp_cross[i0, i1]

    def _init_b_cross(self, nffatype, b_cross, bs, b_mix, b_mix_coeff, amps):
        for i0 in range(nffatype):
            for i1 in range(i0+1):
                if b_cross[i0, i1] == 0.0:
                    if b_mix == 0:
                        b_cross[i0, i1] = (bs[i0] + bs[i1])/2
                    elif amps[i0] == 0.0 or amps[i1] == 0.0:
                        b = 0.0
                    else:
                        b = (bs[i0] + bs[i1])/2;
                        b *= 1 - b_mix_coeff*abs(np.log(amps[i0]/amps[i1]));
                        b_cross[i0, i1] = b
                    b_cross[i1, i0] = b_cross[i0, i1]

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1          A          B')
            log.hline()
            for i0 in range(self._c_nffatype):
                for i1 in range(i0+1):
                    log('%11i %11i %s %s' % (i0, i1, log.energy(self._c_amp_cross[i0, i1]), log.invlength(self._c_b_cross[i0,i1])))

    def _get_amp_cross(self):
        '''The amplitude cross parameters'''
        return self._c_amp_cross.view()

    amp_cross = property(_get_amp_cross)

    def _get_b_cross(self):
        '''The decay cross parameters'''
        return self._c_b_cross.view()

    b_cross = property(_get_b_cross)



cdef class PairPotLJCross(PairPot):
    r'''Lennard Jones with explicit cross parameters.

       **Energy:**

       .. math:: E_\text{LJ} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} 4 s_{ij} \epsilon_{ij} \left[
                 \left(\frac{\sigma_{ij}}{d_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{d_{ij}}\right)^6
                 \right]

       with

       .. math:: s_{ij} = \text{the short-range scaling factor}


       **Arguments:**

        ffatype_ids
            An array with atom type IDs for each atom. The IDs are integer
            indexes for the atom types that start counting from zero. shape =
            (natom,).

       eps_cross
            An array with epsilon parameters, one for each combination of atom types, shape (nffatype,nffatype)

       sig_cross
            An array with epsilon parameters, one for each combination of atom types, shape (nffatype,nffatype)

       rcut
            The cutoff radius

       **Optional arguments:**

       tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied
    '''
    cdef long _c_nffatype
    cdef np.ndarray _c_eps_cross
    cdef np.ndarray _c_sig_cross
    name = 'ljcross'

    def __cinit__(self, np.ndarray[long, ndim=1] ffatype_ids not None,
                  np.ndarray[double, ndim=2] eps_cross not None,
                  np.ndarray[double, ndim=2] sig_cross not None,
                  double rcut, Truncation tr=None):
        assert ffatype_ids.flags['C_CONTIGUOUS']
        assert eps_cross.flags['C_CONTIGUOUS']
        assert sig_cross.flags['C_CONTIGUOUS']
        nffatype = eps_cross.shape[0]
        assert ffatype_ids.min() >= 0
        assert ffatype_ids.max() < nffatype
        assert eps_cross.shape[1] == nffatype
        assert sig_cross.shape[0] == nffatype
        assert sig_cross.shape[1] == nffatype
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_ljcross_init(
            self._c_pair_pot, nffatype, <long*> ffatype_ids.data,
            <double*> eps_cross.data, <double*> sig_cross.data,
        )
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_nffatype = nffatype
        self._c_eps_cross = eps_cross
        self._c_sig_cross = sig_cross

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1         eps          sig')
            log.hline()
            for i0 in range(self._c_nffatype):
                for i1 in range(i0+1):
                    log('%11i %11i %s %s' % (i0, i1, log.energy(self._c_eps_cross[i0,i1]), log.length(self._c_sig_cross[i0,i1])))

    def _get_eps_cross(self):
        '''The C6 cross parameters'''
        return self._c_eps_cross.view()

    eps_cross = property(_get_eps_cross)

    def _get_sig_cross(self):
        '''The damping cross parameters'''
        return self._c_sig_cross.view()

    sig_cross = property(_get_sig_cross)


cdef class PairPotDampDisp(PairPot):
    r'''Damped dispersion interaction

        **Energy:**

        .. math:: E_\text{DAMPDISP} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} s_{ij} C_{n,ij} f_\text{damp,n}(d_{ij}) d_{ij}^{-n}

        where the damping factor :math:`f_\text{damp}(d_{ij})` is optional. When used
        it has the Tang-Toennies form:

        .. math:: f_\text{damp,n}(d_{ij}) = 1 - \exp(-B_{ij}r)\sum_{k=0}^n\frac{(B_{ij}r)^k}{k!}

        The pair parameters :math:`C_{n,ij}` and :math:`B_{ij}` are derived from atomic
        parameters using mixing rules, unless they are provided explicitly for a given
        pair of atom types. These are the mixing rules:

        .. math:: C_{n,ij} = \frac{2 C_{n,i} C_{n,j}}{\left(\frac{V_j}{V_i}\right)^2 C_{n,i} + \left(\frac{V_i}{V_j}\right)^2 C_{n,j}}

        .. math:: B_{ij} = \frac{B_i+B_j}{2}

        **Arguments:**

        ffatype_ids
            An array with atom type IDs for each atom. The IDs are integer
            indexes for the atom types that start counting from zero. shape =
            (natom,).

        cn_cross
            The :math:`C_{n,ij}` cross parameters.

        b_cross
            The :math:`B_{ij}` cross parameters. When zero, the damping factor
            is set to one.

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        cns
            The diagonal :math:`C_{n,i}` parameters

        bs
            The diagonal :math:`B_{i}` parameters

        vols
            The atomic volumes, :math:`V_i`.

        power
            The power to which :math:`1/d_{ij}` is raised.
            Default value is 6, which corresponds to the first term in a
            perturbation expansion of the dispersion energy.

        The three last optional arguments are used to determine pair parameters
        from the mixing rules. These mixing rules are only applied of the
        corresponding cross parameters are initially set to zero in the arrays
        ``cn_cross`` and ``b_cross``.
    '''
    cdef long _c_nffatype
    cdef long _c_power
    cdef np.ndarray _c_cn_cross
    cdef np.ndarray _c_b_cross
    name = 'dampdisp'

    def __cinit__(self, np.ndarray[long, ndim=1] ffatype_ids not None,
                  np.ndarray[double, ndim=2] cn_cross not None,
                  np.ndarray[double, ndim=2] b_cross not None,
                  double rcut, Truncation tr=None,
                  np.ndarray[double, ndim=1] cns=None,
                  np.ndarray[double, ndim=1] bs=None,
                  np.ndarray[double, ndim=1] vols=None,
                  long power=6):
        assert ffatype_ids.flags['C_CONTIGUOUS']
        assert cn_cross.flags['C_CONTIGUOUS']
        assert b_cross.flags['C_CONTIGUOUS']
        nffatype = cn_cross.shape[0]
        assert ffatype_ids.min() >= 0
        assert ffatype_ids.max() < nffatype
        assert cn_cross.shape[1] == nffatype
        assert b_cross.shape[0] == nffatype
        assert b_cross.shape[1] == nffatype
        if cns is not None or vols is not None:
            assert cns is not None
            assert vols is not None
            assert cns.flags['C_CONTIGUOUS']
            assert vols.flags['C_CONTIGUOUS']
            assert cns.shape[0] == nffatype
            assert bs.shape[0] == nffatype
            self._init_cn_cross(nffatype, cn_cross, cns, vols)
        if bs is not None:
            assert bs.flags['C_CONTIGUOUS']
            self._init_b_cross(nffatype, b_cross, bs)
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_dampdisp_init(
            self._c_pair_pot, nffatype, power, <long*> ffatype_ids.data,
            <double*> cn_cross.data, <double*> b_cross.data,
        )
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_nffatype = nffatype
        self._c_power = power
        self._c_cn_cross = cn_cross
        self._c_b_cross = b_cross

    def _init_cn_cross(self, nffatype, cn_cross, cns, vols):
        for i0 in range(nffatype):
            for i1 in range(i0+1):
                if cn_cross[i0, i1] == 0.0 and vols[i0] != 0.0 and vols[i1] != 0.0:
                    ratio = vols[i0]/vols[i1]
                    cn_cross[i0, i1] = 2.0*cns[i0]*cns[i1]/(cns[i0]/ratio+cns[i1]*ratio)
                    cn_cross[i1, i0] = cn_cross[i0, i1]

    def _init_b_cross(self, nffatype, b_cross, bs):
        for i0 in range(nffatype):
            for i1 in range(i0+1):
                if b_cross[i0, i1] == 0.0 and bs[i0] != 0.0 and bs[i1] != 0.0:
                    b_cross[i0, i1] = 0.5*(bs[i0] + bs[i1])
                    b_cross[i1, i0] = b_cross[i0, i1]

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1         cn[au]        B')
            log.hline()
            for i0 in range(self._c_nffatype):
                for i1 in range(i0+1):
                    log('%11i %11i %10.5f %s' % (i0, i1, self._c_cn_cross[i0,i1], log.invlength(self._c_b_cross[i0,i1])))

    def _get_cn_cross(self):
        '''The cn cross parameters'''
        return self._c_cn_cross.view()

    cn_cross = property(_get_cn_cross)

    def _get_b_cross(self):
        '''The damping cross parameters'''
        return self._c_b_cross.view()

    b_cross = property(_get_b_cross)


cdef class PairPotDisp68BJDamp(PairPot):
    r'''Dispersion term with r^-6 and r^-8 term and Becke-Johnson damping

        **Arguments:**

        ffatype_ids
            An array with atom type IDs for each atom. The IDs are integer
            indexes for the atom types that start counting from zero. shape =
            (natom,).

        c6_cross
            The :math:`C_{6,ij}` cross parameters.

        c8_cross
            The :math:`C_{8,ij}` cross parameters.

        R_cross
            The R cross parameters. If not supplied, these are computed as
            sqrt(C8/C6).

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        c6s
            The diagonal :math:`C_{6,i}` parameters

        c8s
            The diagonal :math:`C_{8,i}` parameters

        Rs
            The diagonal parameters

        c6_scale
            Overall scaling of c6 energy. (Default=1.0)

        c8_scale
            Overall scaling of c8 energy. (Default=1.0)

        bj_a
            Parameter to control Becke-Johnson damping.

        bj_b
            Another parameter to control Becke Johnson damping

        The three last optional arguments are used to determine pair parameters
        from the mixing rules. These mixing rules are only applied if the
        corresponding cross parameters are initially set to zero in the arrays
        ``c6_cross``, ``c8_cross`` and ``R_cross``.
    '''
    cdef long _c_nffatype
    cdef np.ndarray _c_c6_cross
    cdef np.ndarray _c_c8_cross
    cdef np.ndarray _c_R_cross
    name = 'disp68bjdamp'

    def __cinit__(self, np.ndarray[long, ndim=1] ffatype_ids not None,
                  np.ndarray[double, ndim=2] c6_cross not None,
                  np.ndarray[double, ndim=2] c8_cross not None,
                  np.ndarray[double, ndim=2] R_cross not None,
                  double rcut, Truncation tr=None,
                  np.ndarray[double, ndim=1] c6s=None,
                  np.ndarray[double, ndim=1] c8s=None,
                  np.ndarray[double, ndim=1] Rs=None,
                  double c6_scale=1.0, double c8_scale=1.0, double bj_a=0.0, double bj_b=0.0):
        assert ffatype_ids.flags['C_CONTIGUOUS']
        assert c6_cross.flags['C_CONTIGUOUS']
        assert c8_cross.flags['C_CONTIGUOUS']
        assert R_cross.flags['C_CONTIGUOUS']
        nffatype = c6_cross.shape[0]
        assert ffatype_ids.min() >= 0
        assert ffatype_ids.max() < nffatype
        assert c6_cross.shape[1] == nffatype
        assert c8_cross.shape[0] == nffatype
        assert c8_cross.shape[1] == nffatype
        assert R_cross.shape[0] == nffatype
        assert R_cross.shape[1] == nffatype
        if c6s is not None:
            assert c6s.flags['C_CONTIGUOUS']
            assert c6s.shape[0] == nffatype
            raise NotImplementedError
        if c8s is not None:
            assert c8s.flags['C_CONTIGUOUS']
            assert c8s.shape[0] == nffatype
            raise NotImplementedError
        if Rs is not None:
            assert Rs.flags['C_CONTIGUOUS']
            assert Rs.shape[0] == nffatype
            raise NotImplementedError
        if np.all(R_cross==0.0):
            # We need to mask out zero c6 coefficients
            mask = c6_cross != 0.0
            #R_cross[mask] = np.sqrt(c8_scale*c8_cross[mask]/c6_cross[mask]/c6_scale)
            R_cross[mask] = np.sqrt(c8_cross[mask]/c6_cross[mask])
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_disp68bjdamp_init(
            self._c_pair_pot, nffatype, <long*> ffatype_ids.data,
            <double*> c6_cross.data, <double*> c8_cross.data,
            <double*> R_cross.data, c6_scale, c8_scale, bj_a, bj_b,
        )
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_nffatype = nffatype
        self._c_c6_cross = c6_cross
        self._c_c8_cross = c8_cross
        self._c_R_cross = R_cross

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_medium:
            log('  c6_scale:             %s' % ("%10.5f"%self.c6_scale))
            log('  c8_scale:             %s' % ("%10.5f"%self.c8_scale))
            log('  bj_a:             %s' % ("%10.5f"%self.bj_a))
            log('  bj_b:             %s' % ("%10.5f"%self.bj_b))
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1         C6         C8          R')
            log.hline()
            for i0 in range(self._c_nffatype):
                for i1 in range(i0+1):
                    log('%11i %11i %s %s %s' % (i0, i1, log.c6(self._c_c6_cross[i0,i1]), "%10.5f"%self._c_c8_cross[i0,i1], log.length(self._c_R_cross[i0,i1])))

    def _get_c6_cross(self):
        '''The C6 cross parameters'''
        return self._c_c6_cross.view()

    c6_cross = property(_get_c6_cross)

    def _get_c8_cross(self):
        '''The C8 cross parameters'''
        return self._c_c8_cross.view()

    c8_cross = property(_get_c8_cross)

    def _get_R_cross(self):
        '''The R cross parameters'''
        return self._c_R_cross.view()

    R_cross = property(_get_R_cross)

    def _get_c6_scale(self):
        '''Global scaling of C6 coefficients'''
        return pair_pot.pair_data_disp68bjdamp_get_c6_scale(self._c_pair_pot)

    c6_scale = property(_get_c6_scale)

    def _get_c8_scale(self):
        '''Global scaling of C8 coefficients'''
        return pair_pot.pair_data_disp68bjdamp_get_c8_scale(self._c_pair_pot)

    c8_scale = property(_get_c8_scale)

    def _get_bj_a(self):
        '''First parameter of Becke-Johnson damping'''
        return pair_pot.pair_data_disp68bjdamp_get_bj_a(self._c_pair_pot)

    bj_a = property(_get_bj_a)

    def _get_bj_b(self):
        '''Global scaling of C6 coefficients'''
        return pair_pot.pair_data_disp68bjdamp_get_bj_b(self._c_pair_pot)

    bj_b = property(_get_bj_b)

    def _get_global_pars(self):
        '''Global parameters'''
        return [self.c6_scale,self.c8_scale,self.bj_a,self.bj_b]

    global_pars = property(_get_global_pars)


cdef class PairPotEI(PairPot):
    r'''Short-range contribution to the electrostatic interaction between point charges

        **Arguments:**

        charges
            An array of atomic charges, shape = (natom,)

        alpha
            The :math:`\alpha` parameter in the Ewald summation scheme. When
            set to zero, the interaction between point charges is computed
            without any long-range screening.

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        dielectric
            A relative dielectric permitivity that just scales the Coulomb
            interaction.

        radii
            An array of atomic radii, shape = (natom,). The charge distribution
            of atom :math:`i` with radius :math:`r_i` centered at :math:`\mathbf{R}_i`
            is of a Gaussian shape:
            :math:`\rho_i (\mathbf{r}) = q_i\left(\frac{1}{\pi r_i^2}\right)^{3/2} \exp{-\frac{|\mathbf{r} -\mathbf{R}_i |^2}{r_i^2}}`
            When the atomic radius equals zero, the charge distribution becomes a
            point monopole.
            Only implemented for non-periodic systems
    '''
    cdef np.ndarray _c_charges
    cdef np.ndarray _c_radii
    name = 'ei'

    def __cinit__(self, np.ndarray[double, ndim=1] charges, double alpha,
                  double rcut, Truncation tr=None, double dielectric=1.0,
                  np.ndarray[double, ndim=1] radii=None):
        assert charges.flags['C_CONTIGUOUS']
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        #No atomic radii specified, set to point charges
        if radii is None: radii = np.zeros( np.shape(charges) )
        pair_pot.pair_data_ei_init(self._c_pair_pot, <double*>charges.data, alpha, dielectric, <double*>radii.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_charges = charges
        self._c_radii = radii

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_medium:
            log('  alpha:                 %s' % log.invlength(self.alpha))
            log('  relative permittivity: %5.3f' % self.dielectric)
        if log.do_high:
            log.hline()
            log('   Atom     Charge     Radius')
            log.hline()
            for i in range(self._c_charges.shape[0]):
                log('%7i %s %s' % (i, log.charge(self._c_charges[i]), log.length(self._c_radii[i])))

    def _get_charges(self):
        '''The atomic charges'''
        return self._c_charges.view()

    charges = property(_get_charges)

    def _get_radii(self):
        '''The atomic radii'''
        return self._c_radii.view()

    radii = property(_get_radii)

    def _get_alpha(self):
        '''The alpha parameter in the Ewald summation method'''
        return pair_pot.pair_data_ei_get_alpha(self._c_pair_pot)

    alpha = property(_get_alpha)

    def _get_dielectric(self):
        '''The scalar relative permittivity'''
        return pair_pot.pair_data_ei_get_dielectric(self._c_pair_pot)

    dielectric = property(_get_dielectric)


cdef class PairPotEIDip(PairPot):
    r'''Short-range contribution to the electrostatic interaction between point charges
        and point dipoles. Only works for non-periodic systems and without truncation scheme

        **Arguments:**

        charges
            An array of atomic charges, shape = (natom,)

        dipoles
            An array of atomic point dipoles, shape = (natom,3)

        poltens_i
            An array that gives the inverse atomic polarizabilities, shape = (3natom, 3)

        TODO: What about other parameters, for example from EEM? More general way
        to include necessary parameters.

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied
    '''
    cdef np.ndarray _c_charges
    cdef np.ndarray _c_dipoles
    cdef np.ndarray _c_poltens_i
    cdef np.ndarray _c_radii
    cdef np.ndarray _c_radii2
    name = 'eidip'

    def __cinit__(self, np.ndarray[double, ndim=1] charges,
                  np.ndarray[double, ndim=2] dipoles, np.ndarray[double, ndim=2] poltens_i,
                    double alpha, double rcut,
                  Truncation tr=None, np.ndarray[double, ndim=1] radii=None, np.ndarray[double, ndim=1] radii2=None):
        assert charges.flags['C_CONTIGUOUS']
        assert dipoles.flags['C_CONTIGUOUS']
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        #No atomic radii specified, set to point charges
        if radii is None: radii = np.zeros( np.shape(charges) )
        #No dipole radii specified, set to point charges
        if radii2 is None: radii2 = np.zeros( np.shape(charges) )
        pair_pot.pair_data_eidip_init(self._c_pair_pot, <double*>charges.data, <double*>dipoles.data, alpha,
                      <double*>radii.data, <double*>radii2.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_charges = charges
        self._c_dipoles = dipoles
        self._c_radii = radii
        self._c_radii2 = radii2
        #Put the polarizability tensors in a matrix with shape that is more convenient for energy calculation
        self._c_poltens_i = np.zeros( (np.shape(poltens_i)[0],np.shape(poltens_i)[0]) )
        for i in range(np.shape(poltens_i)[0]//3):
            self.poltens_i[3*i:3*(i+1) , 3*i:3*(i+1)] = poltens_i[3*i:3*(i+1),:]

    def compute(self, np.ndarray[nlist.neigh_row_type, ndim=1] neighs,
                np.ndarray[pair_pot.scaling_row_type, ndim=1] stab,
                np.ndarray[double, ndim=2] gpos,
                np.ndarray[double, ndim=2] vtens, long nneigh):
        #Override parents method to add dipole creation energy
        #TODO: Does this contribute to gpos or vtens?
        log("Computing PairPotEIDip energy and gradient")
        E = PairPot.compute(self, neighs, stab, gpos, vtens, nneigh)
        E += 0.5*np.dot( np.transpose(np.reshape( self._c_dipoles, (-1,) )) , np.dot( self.poltens_i, np.reshape( self._c_dipoles, (-1,) ) ) )
        return E


    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('Atom     Charge     Radius   Dipole_x   Dipole_y   Dipole_z   Radius')
            log.hline()
            for i in range(self._c_charges.shape[0]):
                log('%4i %s %s %s %s %s %s' % (i, log.charge(self._c_charges[i]),log.length(self._c_radii[i]),\
                log.charge(self._c_dipoles[i,0]), log.charge(self._c_dipoles[i,1]),log.charge(self._c_dipoles[i,2]),\
                log.length(self._c_radii2[i])))

    def _get_charges(self):
        '''The atomic charges'''
        return self._c_charges.view()

    charges = property(_get_charges)

    def _get_dipoles(self):
        '''The atomic charges'''
        return self._c_dipoles.view()

    dipoles = property(_get_dipoles)

    def _get_poltens_i(self):
        '''The atomic charges'''
        return self._c_poltens_i.view()

    poltens_i = property(_get_poltens_i)

    def _get_alpha(self):
        '''The alpha parameter in the Ewald summation method'''
        return pair_pot.pair_data_eidip_get_alpha(self._c_pair_pot)

    alpha = property(_get_alpha)

cdef class PairPotEiSlater1s1sCorr(PairPot):
    r'''Electrostatic interaction between sites with a point core charge and a
        1s Slater charge density MINUS the electrostatic interaction between
        the resulting net point charges.
        TODO: explain this properly

        **Arguments:**

        slater1s_widths
            An array of Slater widths, shape = (natom,)

        slater1s_N
            An array of Slater populations, shape = (natom,)

        slater1s_Z
            An array of effective core charges, shape = (natom,)

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied
    '''
    cdef np.ndarray _c_slater1s_widths
    cdef np.ndarray _c_slater1s_N
    cdef np.ndarray _c_slater1s_Z
    name = 'eislater1s1scorr'

    def __cinit__(self, np.ndarray[double, ndim=1] slater1s_widths,
                  np.ndarray[double, ndim=1] slater1s_N, np.ndarray[double, ndim=1] slater1s_Z,
                  double rcut, Truncation tr=None):
        assert slater1s_widths.flags['C_CONTIGUOUS']
        assert slater1s_N.flags['C_CONTIGUOUS']
        assert slater1s_Z.flags['C_CONTIGUOUS']
        # Precompute some factors here???
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_eislater1s1scorr_init(self._c_pair_pot, <double*>slater1s_widths.data,  <double*>slater1s_N.data,  <double*>slater1s_Z.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_slater1s_widths = slater1s_widths
        self._c_slater1s_N = slater1s_N
        self._c_slater1s_Z = slater1s_Z

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('   Atom  Slater charge  Core charge   Slater width')
            log.hline()
            for i in range(self._c_slater1s_widths.shape[0]):
                log('%7i     %s   %s     %s' % (i, log.charge(self._c_slater1s_N[i]),log.charge(self._c_slater1s_Z[i]),log.length(self._c_slater1s_widths[i])))

    def _get_slater1s_widths(self):
        '''The atomic charges'''
        return self._c_slater1s_widths.view()

    slater1s_widths = property(_get_slater1s_widths)

    def _get_slater1s_N(self):
        '''The atomic charges'''
        return self._c_slater1s_N.view()

    slater1s_N = property(_get_slater1s_N)

    def _get_slater1s_Z(self):
        '''The atomic charges'''
        return self._c_slater1s_Z.view()

    slater1s_Z = property(_get_slater1s_Z)


cdef class PairPotEiSlater1sp1spCorr(PairPot):
    r'''Electrostatic interaction between sites with a point charge, a
        1s Slater charge density, a point dipole and a 1p Slater charge density
        MINUS the electrostatic interaction between
        the resulting net point monopoles and dipoles.
        TODO: explain this properly

        **Arguments:**

        slater1s_widths
            An array of Slater widths, shape = (natom,)

        slater1s_N
            An array of Slater populations, shape = (natom,)

        slater1s_Z
            An array of effective core charges, shape = (natom,)

        slater1p_widths
            An array of Slater widths, shape = (natom,3)

        slater1p_N
            An array of Slater populations, shape = (natom,3)

        slater1p_Z
            An array of point dipoles, shape = (natom,3)

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied
    '''
    cdef np.ndarray _c_slater1s_widths
    cdef np.ndarray _c_slater1s_N
    cdef np.ndarray _c_slater1s_Z
    cdef np.ndarray _c_slater1p_widths
    cdef np.ndarray _c_slater1p_N
    cdef np.ndarray _c_slater1p_Z
    name = 'eislater1sp1spcorr'

    def __cinit__(self, np.ndarray[double, ndim=1] slater1s_widths,
                  np.ndarray[double, ndim=1] slater1s_N, np.ndarray[double, ndim=1] slater1s_Z,
                  np.ndarray[double, ndim=2] slater1p_widths, np.ndarray[double, ndim=2] slater1p_N,
                  np.ndarray[double, ndim=2] slater1p_Z, double rcut, Truncation tr=None):
        assert slater1s_widths.flags['C_CONTIGUOUS']
        assert slater1s_N.flags['C_CONTIGUOUS']
        assert slater1s_Z.flags['C_CONTIGUOUS']
        assert slater1p_widths.flags['C_CONTIGUOUS']
        assert slater1p_N.flags['C_CONTIGUOUS']
        assert slater1p_Z.flags['C_CONTIGUOUS']
        # Precompute some factors here???
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_eislater1sp1spcorr_init(self._c_pair_pot, <double*>slater1s_widths.data,  <double*>slater1s_N.data,  <double*>slater1s_Z.data,
                                                   <double*>slater1p_widths.data, <double*>slater1p_N.data, <double*>slater1p_Z.data)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_slater1s_widths = slater1s_widths
        self._c_slater1s_N = slater1s_N
        self._c_slater1s_Z = slater1s_Z
        self._c_slater1p_widths = slater1p_widths
        self._c_slater1p_N = slater1p_N
        self._c_slater1p_Z = slater1p_Z


    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_high:
            log.hline()
            log('   Atom  Slater charge  Core charge   Slater width')
            log.hline()
            for i in range(self._c_slater1s_widths.shape[0]):
                log('%7i     %s   %s     %s' % (i, log.charge(self._c_slater1s_N[i]),log.charge(self._c_slater1s_Z[i]),log.length(self._c_slater1s_widths[i])))

    def _get_slater1s_widths(self):
        '''The atomic charges'''
        return self._c_slater1s_widths.view()

    slater1s_widths = property(_get_slater1s_widths)

    def _get_slater1s_N(self):
        '''The atomic charges'''
        return self._c_slater1s_N.view()

    slater1s_N = property(_get_slater1s_N)

    def _get_slater1s_Z(self):
        '''The atomic charges'''
        return self._c_slater1s_Z.view()

    slater1s_Z = property(_get_slater1s_Z)

    def _get_slater1p_widths(self):
        '''The atomic charges'''
        return self._c_slater1p_widths.view()

    slater1p_widths = property(_get_slater1p_widths)

    def _get_slater1p_N(self):
        '''The atomic charges'''
        return self._c_slater1p_N.view()

    slater1p_N = property(_get_slater1p_N)

    def _get_slater1p_Z(self):
        '''The atomic charges'''
        return self._c_slater1p_Z.view()

    slater1p_Z = property(_get_slater1p_Z)


cdef class PairPotOlpSlater1s1s(PairPot):
    r'''Overlap between two Slater 1s densities. This can for instance be used
        to represent an exchange energy by defining a suitable scaling factor
        to convert overlap to energy. Furthermore it is possible to add some
        correction factors defined in following expression:

            E = ex_scale * slater_overlap * (1+corr_c*(N1+N2))*(1-exp(corr_a-corr_b*r/sqrt(sigma1*sigma2)))

        **Arguments:**

        slater1s_widths
            An array of Slater widths, shape = (natom,)

        slater1s_N
            An array of Slater populations, shape = (natom,)

        ex_scale
            A scaling factor to relate overlap and exchange energy

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        corr_a
            Correction factor to modify overlap expression (default=0.0)

        corr_b
            Correction factor to modify overlap expression (default=0.0)

        corr_c
            Correction factor to modify overlap expression (default=0.0)
    '''
    cdef np.ndarray _c_slater1s_widths
    cdef np.ndarray _c_slater1s_N
    name = 'olpslater1s1s'

    def __cinit__(self, np.ndarray[double, ndim=1] slater1s_widths,
                  np.ndarray[double, ndim=1] slater1s_N, double ex_scale,
                  double rcut, Truncation tr=None, double corr_a=0.0,
                  double corr_b=0.0, double corr_c=0.0):
        assert slater1s_widths.flags['C_CONTIGUOUS']
        assert slater1s_N.flags['C_CONTIGUOUS']
        # Precompute some factors here???
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_olpslater1s1s_init(self._c_pair_pot, <double*>slater1s_widths.data,  <double*>slater1s_N.data,  ex_scale, corr_a, corr_b, corr_c)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_slater1s_widths = slater1s_widths
        self._c_slater1s_N = slater1s_N

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_medium:
            log('  ex_scale:             %s' % ("%10.5f"%self.ex_scale))
            log('  corr_a:             %s' % ("%10.5f"%self.corr_a))
            log('  corr_b:             %s' % ("%10.5f"%self.corr_b))
            log('  corr_c:             %s' % ("%10.5f"%self.corr_c))
        if log.do_high:
            log.hline()
            log('   Atom  Slater charge   Slater width')
            log.hline()
            for i in range(self._c_slater1s_widths.shape[0]):
                log('%7i     %s     %s' % (i, log.charge(self._c_slater1s_N[i]),log.length(self._c_slater1s_widths[i])))

    def _get_slater1s_widths(self):
        '''The atomic charges'''
        return self._c_slater1s_widths.view()

    slater1s_widths = property(_get_slater1s_widths)

    def _get_slater1s_N(self):
        '''The atomic charges'''
        return self._c_slater1s_N.view()

    slater1s_N = property(_get_slater1s_N)

    def _get_ex_scale(self):
        '''The ex_scale parameter in the exchange energy expression'''
        return pair_pot.pair_data_olpslater1s1s_get_ex_scale(self._c_pair_pot)

    ex_scale = property(_get_ex_scale)

    def _get_corr_a(self):
        '''The corr_a parameter in the exchange energy expression'''
        return pair_pot.pair_data_olpslater1s1s_get_corr_a(self._c_pair_pot)

    corr_a = property(_get_corr_a)

    def _get_corr_b(self):
        '''The corr_b parameter in the exchange energy expression'''
        return pair_pot.pair_data_olpslater1s1s_get_corr_b(self._c_pair_pot)

    corr_b = property(_get_corr_b)

    def _get_corr_c(self):
        '''The corr_c parameter in the exchange energy expression'''
        return pair_pot.pair_data_olpslater1s1s_get_corr_c(self._c_pair_pot)

    corr_c = property(_get_corr_c)


cdef class PairPotChargeTransferSlater1s1s(PairPot):
    r'''Model for charge transfer energy proportional to the overlap of two 1s
        Slater densities and a certain power of the product of Slater widths:

            E = ct_scale * slater_overlap / (sigma1*sigma2)**width_power

        **Arguments:**

        slater1s_widths
            An array of Slater widths, shape = (natom,)

        slater1s_N
            An array of Slater populations, shape = (natom,)

        ct_scale
            A scaling factor to relate overlap and exchange energy

        rcut
            The cutoff radius

        **Optional arguments:**

        tr
            The truncation scheme, an instance of a subclass of ``Truncation``.
            When not given, no truncation is applied

        width_power
            Correction factor to modify overlap expression (default=3.0)

    '''
    cdef np.ndarray _c_slater1s_widths
    cdef np.ndarray _c_slater1s_N
    name = 'chargetransferslater1s1s'

    def __cinit__(self, np.ndarray[double, ndim=1] slater1s_widths,
                  np.ndarray[double, ndim=1] slater1s_N, double ct_scale,
                  double rcut, Truncation tr=None, double width_power=3.0):
        assert slater1s_widths.flags['C_CONTIGUOUS']
        assert slater1s_N.flags['C_CONTIGUOUS']
        # Precompute some factors here???
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_chargetransferslater1s1s_init(self._c_pair_pot, <double*>slater1s_widths.data,  <double*>slater1s_N.data, ct_scale, width_power)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_slater1s_widths = slater1s_widths
        self._c_slater1s_N = slater1s_N

    def log(self):
        '''Print suitable initialization info on screen.'''
        if log.do_medium:
            log('  ct_scale:             %s' % ("%10.5f"%self.ct_scale))
            log('  width_power:             %s' % ("%10.5f"%self.width_power))
        if log.do_high:
            log.hline()
            log('   Atom  Slater charge   Slater width')
            log.hline()
            for i in range(self._c_slater1s_widths.shape[0]):
                log('%7i     %s     %s' % (i, log.charge(self._c_slater1s_N[i]),log.length(self._c_slater1s_widths[i])))

    def _get_slater1s_widths(self):
        '''The atomic charges'''
        return self._c_slater1s_widths.view()

    slater1s_widths = property(_get_slater1s_widths)

    def _get_slater1s_N(self):
        '''The atomic charges'''
        return self._c_slater1s_N.view()

    slater1s_N = property(_get_slater1s_N)

    def _get_ct_scale(self):
        '''The ct_scale parameter in the charge transfer energy expression'''
        return pair_pot.pair_data_chargetransferslater1s1s_get_ct_scale(self._c_pair_pot)

    ct_scale = property(_get_ct_scale)

    def _get_width_power(self):
        '''The corr_a parameter in the charge transfer energy expression'''
        return pair_pot.pair_data_chargetransferslater1s1s_get_width_power(self._c_pair_pot)

    width_power = property(_get_width_power)



#
# Ewald summation stuff
#


def compute_ewald_reci(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       Cell unitcell, double alpha,
                       np.ndarray[long, ndim=1] gmax,
                       double gcut, double dielectric,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=1] work,
                       np.ndarray[double, ndim=2] vtens):
    '''Compute the reciprocal interaction term in the Ewald summation scheme

       **Arguments:**

       pos
            The atomic positions. numpy array with shape (natom,3).

       charges
            The atomic charges. numpy array with shape (natom,).

       unitcell
            An instance of the ``Cell`` class that describes the periodic
            boundary conditions.

       alpha
            The :math:`\\alpha` parameter from the Ewald summation scheme.

       gmax
            The maximum range of periodic images in reciprocal space to be
            considered for the Ewald sum. integer numpy array with shape (3,).
            Each element gives the range along the corresponding reciprocal
            cell vector. The range along each axis goes from -gmax[0] to
            gmax[0] (inclusive).

       gcut
            The cutoff in reciprocal space. The caller is responsible for the
            compatibility of ``gcut`` with ``gmax``.

       dielectric
            The scalar relative permittivity of the system.

       gpos
            If not set to None, the Cartesian gradient of the energy is
            stored in this array. numpy array with shape (natom, 3).

       work
            If gpos is given, this work array must also be present. Its
            contents will be overwritten. numpy array with shape (2*natom,).

       vtens
            If not set to None, the virial tensor is computed and stored in
            this array. numpy array with shape (3, 3).
    '''
    cdef double *my_gpos
    cdef double *my_work
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert unitcell.nvec == 3
    assert alpha > 0
    assert dielectric >= 1.0
    assert gmax.flags['C_CONTIGUOUS']
    assert gmax.shape[0] == 3

    if gpos is None:
        my_gpos = NULL
        my_work = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        assert gpos.shape[0] == pos.shape[0]
        assert work.flags['C_CONTIGUOUS']
        assert gpos.shape[0]*2 == work.shape[0]
        my_gpos = <double*>gpos.data
        my_work = <double*>work.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    return ewald.compute_ewald_reci(<double*>pos.data, len(pos),
                                    <double*>charges.data,
                                    unitcell._c_cell, alpha, <long*>gmax.data,
                                    gcut, dielectric, my_gpos, my_work,
                                    my_vtens)


def compute_ewald_reci_dd(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       np.ndarray[double, ndim=2] dipoles,
                       Cell unitcell, double alpha,
                       np.ndarray[long, ndim=1] gmax, double gcut,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=1] work,
                       np.ndarray[double, ndim=2] vtens):
    '''Compute the reciprocal interaction term in the Ewald summation scheme

       **Arguments:**

       pos
            The atomic positions. numpy array with shape (natom,3).

       charges
            The atomic charges. numpy array with shape (natom,).

       dipoles
            The atomic dipoles. numpy array with shape (natom,3).

       unitcell
            An instance of the ``Cell`` class that describes the periodic
            boundary conditions.

       alpha
            The :math:`\\alpha` parameter from the Ewald summation scheme.

       gmax
            The maximum range of periodic images in reciprocal space to be
            considered for the Ewald sum. integer numpy array with shape (3,).
            Each element gives the range along the corresponding reciprocal
            cell vector. The range along each axis goes from -gmax[0] to
            gmax[0] (inclusive).

       gcut
            The cutoff in reciprocal space. The caller is responsible for the
            compatibility of ``gcut`` with ``gmax``.

       gpos
            If not set to None, the Cartesian gradient of the energy is
            stored in this array. numpy array with shape (natom, 3).

       work
            If gpos is given, this work array must also be present. Its
            contents will be overwritten. numpy array with shape (2*natom,).

       vtens
            If not set to None, the virial tensor is computed and stored in
            this array. numpy array with shape (3, 3).
    '''
    cdef double *my_gpos
    cdef double *my_work
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert dipoles.flags['C_CONTIGUOUS']
    assert dipoles.shape[0] == pos.shape[0]
    assert unitcell.nvec == 3
    assert alpha > 0
    assert gmax.flags['C_CONTIGUOUS']
    assert gmax.shape[0] == 3

    if gpos is None:
        my_gpos = NULL
        my_work = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        assert gpos.shape[0] == pos.shape[0]
        assert work.flags['C_CONTIGUOUS']
        assert gpos.shape[0]*2 == work.shape[0]
        my_gpos = <double*>gpos.data
        my_work = <double*>work.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    return ewald.compute_ewald_reci_dd(<double*>pos.data, len(pos),
                                    <double*>charges.data,
                                    <double*>dipoles.data,
                                    unitcell._c_cell, alpha,
                                    <long*>gmax.data, gcut, my_gpos, my_work,
                                    my_vtens)


def compute_ewald_corr(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       Cell unitcell, double alpha,
                       np.ndarray[pair_pot.scaling_row_type, ndim=1] stab,
                       double dielectric,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=2] vtens):
    '''Compute the corrections to the reciprocal Ewald term due to scaled
       short-range non-bonding interactions.

       **Arguments:**

       pos
            The atomic positions. numpy array with shape (natom,3).

       charges
            The atomic charges. numpy array with shape (natom,).

       unitcell
            An instance of the ``Cell`` class that describes the periodic
            boundary conditions.

       alpha
            The :math:`\\alpha` parameter from the Ewald summation scheme.

       stab
            The table with (sorted) pairs of atoms whose electrostatic
            interactions are scaled. Each record corresponds to one pair
            and contains the corresponding amount of scaling. See
            ``pair_pot.scaling_row_type``

       dielectric
            The scalar relative permittivity of the system.

       gpos
            If not set to None, the Cartesian gradient of the energy is
            stored in this array. numpy array with shape (natom, 3).

       vtens
            If not set to None, the virial tensor is computed and stored in
            this array. numpy array with shape (3, 3).
    '''

    cdef double *my_gpos
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert alpha > 0
    assert stab.flags['C_CONTIGUOUS']

    if gpos is None:
        my_gpos = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        assert gpos.shape[0] == pos.shape[0]
        my_gpos = <double*>gpos.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    return ewald.compute_ewald_corr(
        <double*>pos.data, <double*>charges.data, unitcell._c_cell, alpha,
        <pair_pot.scaling_row_type*>stab.data, len(stab), dielectric,
        my_gpos, my_vtens, len(pos)
    )

def compute_ewald_corr_dd(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       np.ndarray[double, ndim=2] dipoles,
                       Cell unitcell, double alpha,
                       np.ndarray[pair_pot.scaling_row_type, ndim=1] stab,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=2] vtens):
    '''Compute the corrections to the reciprocal Ewald term due to scaled
       short-range non-bonding interactions.

       **Arguments:**

       pos
            The atomic positions. numpy array with shape (natom,3).

       charges
            The atomic charges. numpy array with shape (natom,).

       unitcell
            An instance of the ``Cell`` class that describes the periodic
            boundary conditions.

       alpha
            The :math:`\\alpha` parameter from the Ewald summation scheme.

       stab
            The table with (sorted) pairs of atoms whose electrostatic
            interactions are scaled. Each record corresponds to one pair
            and contains the corresponding amount of scaling. See
            ``pair_pot.scaling_row_type``

       gpos
            If not set to None, the Cartesian gradient of the energy is
            stored in this array. numpy array with shape (natom, 3).

       vtens
            If not set to None, the virial tensor is computed and stored in
            this array. numpy array with shape (3, 3).
    '''

    cdef double *my_gpos
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert dipoles.flags['C_CONTIGUOUS']
    assert dipoles.shape[0] == pos.shape[0]
    assert dipoles.shape[1] == pos.shape[1]
    assert alpha > 0
    assert stab.flags['C_CONTIGUOUS']

    if gpos is None:
        my_gpos = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        assert gpos.shape[0] == pos.shape[0]
        my_gpos = <double*>gpos.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    return ewald.compute_ewald_corr_dd(
        <double*>pos.data, <double*>charges.data, <double*>dipoles.data, unitcell._c_cell, alpha,
        <pair_pot.scaling_row_type*>stab.data, len(stab), my_gpos,
        my_vtens, len(pos)
    )


#
# Delta list
#


def dlist_forward(np.ndarray[double, ndim=2] pos,
                  Cell unitcell,
                  np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
    '''Compute the relative vectors in the delta list

       **Arguments:**

       pos
            The atomic positions. numpy array with shape (natom,3).

       unitcell
            An instance of the ``Cell`` class that describes the periodic
            boundary conditions.

       deltas
            The delta list array

       ndelta
            The number of records in the delta list that need to be computed.
    '''
    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert deltas.flags['C_CONTIGUOUS']
    dlist.dlist_forward(<double*>pos.data, unitcell._c_cell,
                        <dlist.dlist_row_type*>deltas.data, ndelta)

def dlist_back(np.ndarray[double, ndim=2] gpos,
               np.ndarray[double, ndim=2] vtens,
               np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
    '''The back-propagation step of the delta list

       **Arguments:**

       gpos
            If not set to None, the Cartesian gradient of the energy is
            stored in this array. numpy array with shape (natom, 3).

       vtens
            If not set to None, the virial tensor is computed and stored in
            this array. numpy array with shape (3, 3).

       deltas
            The delta list array

       ndelta
            The number of records in the delta list that need to be computed.
    '''
    cdef double *my_gpos
    cdef double *my_vtens

    assert deltas.flags['C_CONTIGUOUS']
    if gpos is None and vtens is None:
        raise TypeError('Either gpos or vtens must be given.')

    if gpos is None:
        my_gpos = NULL
    else:
        assert gpos.flags['C_CONTIGUOUS']
        assert gpos.shape[1] == 3
        my_gpos = <double*>gpos.data

    if vtens is None:
        my_vtens = NULL
    else:
        assert vtens.flags['C_CONTIGUOUS']
        assert vtens.shape[0] == 3
        assert vtens.shape[1] == 3
        my_vtens = <double*>vtens.data

    dlist.dlist_back(my_gpos, my_vtens,
                     <dlist.dlist_row_type*>deltas.data, ndelta)


#
# InternalCoordinate list
#


def iclist_forward(np.ndarray[dlist.dlist_row_type, ndim=1] deltas,
                   np.ndarray[iclist.iclist_row_type, ndim=1] ictab, long nic):
    '''Compute internal coordinates based on relative vectors

       **Arguments:**

       deltas
            The delta list array (input)

       ictab
            The table with internal coordinates that must be computed (input and
            output).

       nic
            The number of records in the ``ictab`` array to compute.
    '''
    assert deltas.flags['C_CONTIGUOUS']
    assert ictab.flags['C_CONTIGUOUS']
    iclist.iclist_forward(<dlist.dlist_row_type*>deltas.data,
                          <iclist.iclist_row_type*>ictab.data, nic)

def iclist_back(np.ndarray[dlist.dlist_row_type, ndim=1] deltas,
                np.ndarray[iclist.iclist_row_type, ndim=1] ictab, long nic):
    '''The back-propagation step in the internal coordinate list

       deltas
            The delta list array (output)

       ictab
            The table with internal coordinates that must be computed (input).

       nic
            The number of records in the ``ictab`` array to compute.

       This routine transforms the partial derivatives of the energy towards the
       internal coordinates, stored in ``ictab``, into partial derivatives of
       the energy towards relative vectors, added to ``deltas``.
    '''
    assert deltas.flags['C_CONTIGUOUS']
    assert ictab.flags['C_CONTIGUOUS']
    iclist.iclist_back(<dlist.dlist_row_type*>deltas.data,
                       <iclist.iclist_row_type*>ictab.data, nic)


#
# Valence list
#


def vlist_forward(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
                  np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    '''Computes valence energy terms based on a list of internal coordinates

       **Arguments:**

       ictab
            The table with internal coordinates (input).

       vtab
            The table with covalent energy terms (input and output).

       nv
            The number of records to consider in ``vtab``.
    '''
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    return vlist.vlist_forward(<iclist.iclist_row_type*>ictab.data,
                               <vlist.vlist_row_type*>vtab.data, nv)

def vlist_back(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
               np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    '''The back-propagation step in the valence list.

       **Arguments:**

       ictab
            The table with internal coordinates (output).

       vtab
            The table with covalent energy terms (input).

       nv
            The number of records to consider in ``vtab``.

       This routine computes the derivatives of the energy of each term towards
       the internal coordinates and adds the results to the ``ictab`` array.
    '''
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    vlist.vlist_back(<iclist.iclist_row_type*>ictab.data,
                     <vlist.vlist_row_type*>vtab.data, nv)

#
# grid
#

def compute_grid3d(np.ndarray[double, ndim=1] center, Cell unitcell, np.ndarray[double, ndim=3] egrid):
    assert center.flags['C_CONTIGUOUS']
    assert center.shape[0] == 3
    return grid.compute_grid3d(<double*>center.data, unitcell._c_cell, <double*>egrid.data, <long*>egrid.shape)
