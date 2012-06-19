# YAFF is yet another force-field code
# Copyright (C) 2008 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
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
cimport numpy as np
cimport cell
cimport nlist
cimport pair_pot
cimport ewald
cimport dlist
cimport iclist
cimport vlist
cimport truncation

from yaff.log import log


__all__ = [
    'Cell', 'nlist_status_init', 'nlist_build', 'nlist_status_finish',
    'nlist_recompute', 'nlist_inc_r', 'Hammer', 'Switch3', 'PairPot',
    'PairPotLJ', 'PairPotMM3', 'PairPotGrimme', 'PairPotExpRep',
    'PairPotDampDisp', 'PairPotEI', 'compute_ewald_reci', 'compute_ewald_corr',
    'dlist_forward', 'dlist_back', 'iclist_forward', 'iclist_back',
    'vlist_forward', 'vlist_back',
]


#
# Cell
#

cdef class Cell:
    cdef cell.cell_type* _c_cell

    def __cinit__(self, *args, **kwargs):
        self._c_cell = cell.cell_new()
        if self._c_cell is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_cell is not NULL:
            cell.cell_free(self._c_cell)

    def __init__(self, np.ndarray[double, ndim=2] rvecs):
        self.update_rvecs(rvecs)

    def update_rvecs(self, np.ndarray[double, ndim=2] rvecs):
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

    def get_nvec(self):
        return cell.cell_get_nvec(self._c_cell)

    nvec = property(get_nvec)

    def get_volume(self):
        return cell.cell_get_volume(self._c_cell)

    volume = property(get_volume)

    def get_rvecs(self, full=False):
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_rvecs(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rvecs = property(get_rvecs)

    def get_gvecs(self, full=False):
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_gvecs(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gvecs = property(get_gvecs)

    def get_rspacings(self, full=False):
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_rspacings(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rspacings = property(get_rspacings)

    def get_gspacings(self, full=False):
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_gspacings(self._c_cell, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gspacings = property(get_gspacings)

    def get_parameters(self):
        rvecs = self.get_rvecs()
        tmp = np.dot(rvecs, rvecs.T)
        lengths = np.sqrt(np.diag(tmp))
        tmp /= lengths
        tmp /= lengths.reshape((-1,1))
        if len(rvecs) < 2:
            cosines = np.arrays([])
        elif len(rvecs) == 2:
            cosines = np.array([tmp[0,1]])
        else:
            cosines = np.array([tmp[1,2], tmp[2,0], tmp[0,1]])
        angles = np.arccos(np.clip(cosines, -1, 1))
        return lengths, angles

    parameters = property(get_parameters)

    def mic(self, np.ndarray[double, ndim=1] delta):
        """Apply the minimum image convention to delta in-place"""
        assert delta.size == 3
        cell.cell_mic(<double*> delta.data, self._c_cell)

    def to_center(self, np.ndarray[double, ndim=1] delta):
        assert delta.size == 3
        cdef np.ndarray[long, ndim=1] result
        result = np.zeros(self.nvec, int)
        cell.cell_to_center(<double*> delta.data, self._c_cell, <long*> result.data)
        return result

    def add_vec(self, np.ndarray[double, ndim=1] delta, np.ndarray[long, ndim=1] r):
        """Add a linear combination of cell vectors in-place"""
        assert delta.size == 3
        assert r.size == self.nvec
        cell.cell_add_vec(<double*> delta.data, self._c_cell, <long*> r.data)

    def compute_distances(self, np.ndarray[double, ndim=1] output,
                          np.ndarray[double, ndim=2] pos0,
                          np.ndarray[double, ndim=2] pos1=None,
                          np.ndarray[long, ndim=2] exclude=None):
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

           exclude
                A sorted array of atom pairs that will be excluded from the
                fitting procedure. The indexes in this array refer to rows of
                pos0 or pos1. If pos1 is not given, both columns refer to rows
                of pos0. If pos1 is given, the first column refers to rows of
                pos0 and the second column refers to rows of pos1. The rows in
                the exclude array should be sorted lexicographically, first
                along the first column, then along the second column.

           This routine can operate in two different ways, depending on the
           presence/absence of the argument ``pos1``. If not given, all
           distances between points in ``pos0`` are computed and the length of
           the output array is ``len(pos0)*(len(pos0)-1)/2``. If ``pos1`` is
           given, all distances are computed between a point in ``pos0`` and a
           point in ``pos1`` and the length of the output array is
           ``len(pos0)*len(pos1)``. In both cases, some pairs of atoms may be
           excluded from the output with the ``exclude`` argument. In typical
           cases, this list of excluded pairs is relatively short.
        """
        cdef long* exclude_pointer

        assert pos0.shape[1] == 3
        assert pos0.flags['C_CONTIGUOUS']
        natom0 = pos0.shape[0]

        if exclude is not None:
            assert exclude.shape[1] == 2
            assert exclude.flags['C_CONTIGUOUS']
            exclude_pointer = <long*> exclude.data
            nexclude = exclude.shape[0]
        else:
            exclude_pointer = NULL
            nexclude = 0

        if pos1 is None:
            if exclude is None:
                assert (natom0*(natom0-1))/2 == output.shape[0]
            else:
                assert (natom0*(natom0-1))/2 - len(exclude) == output.shape[0]
            if cell.is_invalid_exclude(<long*> exclude.data, natom0, natom0, nexclude, True):
                raise ValueError('The exclude array must countain indices within proper bounds ans must be lexicographically sorted.')
            cell.cell_compute_distances1(self._c_cell, <double*> pos0.data,
                                         <double*> output.data, natom0,
                                         <long*> exclude_pointer, nexclude)
        else:
            assert pos1.shape[1] == 3
            assert pos1.flags['C_CONTIGUOUS']
            natom1 = pos1.shape[0]

            if exclude is None:
                assert natom0*natom1 == output.shape[0]
            else:
                assert natom0*natom1 - len(exclude) == output.shape[0]
            if cell.is_invalid_exclude(<long*> exclude.data, natom0, natom1, nexclude, False):
                raise ValueError('The exclude array must countain indices within proper bounds ans must be lexicographically sorted.')
            cell.cell_compute_distances2(self._c_cell, <double*> pos0.data,
                                         <double*> pos1.data,
                                         <double*> output.data, natom0, natom1,
                                         <long*> exclude_pointer, nexclude)


#
# Neighbor lists
#


def nlist_status_init(rmax):
    # seven integer status fields:
    # * r0
    # * r1
    # * r2
    # * a
    # * b
    # * sign
    # * number of rows consumed
    result = np.array([0, 0, 0, 0, 0, 1, 0], int)
    return result


def nlist_build(np.ndarray[double, ndim=2] pos, double rcut,
                np.ndarray[long, ndim=1] rmax,
                Cell unitcell, np.ndarray[long, ndim=1] status,
                np.ndarray[nlist.neigh_row_type, ndim=1] neighs):
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
    return status[-1]


def nlist_recompute(np.ndarray[double, ndim=2] pos,
                    np.ndarray[double, ndim=2] pos_old,
                    Cell unitcell,
                    np.ndarray[nlist.neigh_row_type, ndim=1] neighs):
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
    return nlist.nlist_inc_r(unitcell._c_cell, <long*>r.data, <long*>rmax.data)


#
# Pair potential truncation schemes
#


cdef class Truncation:
    cdef truncation.trunc_scheme_type* _c_trunc_scheme

    def __dealloc__(self):
        if self._c_trunc_scheme is not NULL:
            truncation.trunc_scheme_free(self._c_trunc_scheme)

    def trunc_fn(self, double d, double rcut):
        cdef double hg
        hg = 0.0
        h = truncation.trunc_scheme_fn(self._c_trunc_scheme, d, rcut, &hg)
        return h, hg


cdef class Hammer(Truncation):
    def __cinit__(self, double tau):
        self._c_trunc_scheme = truncation.hammer_new(tau)
        if self._c_trunc_scheme is NULL:
            raise MemoryError

    def get_tau(self):
        return truncation.hammer_get_tau(self._c_trunc_scheme)

    tau = property(get_tau)

    def get_log(self):
        return 'hammer %s' % log.length(self.tau)


cdef class Switch3(Truncation):
    def __cinit__(self, double width):
        self._c_trunc_scheme = truncation.switch3_new(width)
        if self._c_trunc_scheme is NULL:
            raise MemoryError

    def get_width(self):
        return truncation.switch3_get_width(self._c_trunc_scheme)

    width = property(get_width)

    def get_log(self):
        return 'switch3 %s' % log.length(self.width)


#
# Pair potentials
#


cdef class PairPot:
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

    def get_rcut(self):
        return pair_pot.pair_pot_get_rcut(self._c_pair_pot)

    rcut = property(get_rcut)

    cdef set_truncation(self, Truncation tr):
        self.tr = tr
        if tr is None:
            pair_pot.pair_pot_set_trunc_scheme(self._c_pair_pot, NULL)
        else:
            pair_pot.pair_pot_set_trunc_scheme(self._c_pair_pot, tr._c_trunc_scheme)

    def get_truncation(self):
        return self.tr

    def compute(self, np.ndarray[nlist.neigh_row_type, ndim=1] neighs,
                np.ndarray[pair_pot.scaling_row_type, ndim=1] stab,
                np.ndarray[double, ndim=2] gpos,
                np.ndarray[double, ndim=2] vtens, long nneigh):
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
        if log.do_high:
            log.hline()
            log('   Atom      Sigma    Epsilon')
            log.hline()
            for i in xrange(self._c_sigmas.shape[0]):
                log('%7i %s %s' % (i, log.length(self._c_sigmas[i]), log.energy(self._c_epsilons[i])))

    def get_sigmas(self):
        return self._c_sigmas.view()

    sigmas = property(get_sigmas)

    def get_epsilons(self):
        return self._c_epsilons.view()

    epsilons = property(get_epsilons)


cdef class PairPotMM3(PairPot):
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
        if log.do_high:
            log.hline()
            log('   Atom      Sigma    Epsilon    OnlyPauli')
            log.hline()
            for i in xrange(self._c_sigmas.shape[0]):
                log('%7i %s %s            %i' % (i, log.length(self._c_sigmas[i]), log.energy(self._c_epsilons[i]), self._c_onlypaulis[i]))

    def get_sigmas(self):
        return self._c_sigmas.view()

    sigmas = property(get_sigmas)

    def get_epsilons(self):
        return self._c_epsilons.view()

    epsilons = property(get_epsilons)

    def get_onlypaulis(self):
        return self._c_onlypaulis.view()

    onlypaulis = property(get_onlypaulis)


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
            for i in xrange(self._c_r0.shape[0]):
                log('%7i %s %s' % (i, log.length(self._c_r0[i]), log.c6(self._c_c6[i])))

    def get_r0(self):
        return self._c_r0.view()

    r0 = property(get_r0)

    def get_c6(self):
        return self._c_c6.view()

    c6 = property(get_c6)


cdef class PairPotExpRep(PairPot):
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
        for i0 in xrange(nffatype):
            for i1 in xrange(i0+1):
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
        for i0 in xrange(nffatype):
            for i1 in xrange(i0+1):
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
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1          A          B')
            log.hline()
            for i0 in xrange(self._c_nffatype):
                for i1 in xrange(i0+1):
                    log('%11i %11i %s %s' % (i0, i1, log.energy(self._c_amp_cross[i0, i1]), log.invlength(self._c_b_cross[i0,i1])))

    def get_amp_cross(self):
        return self._c_amp_cross.view()

    amp_cross = property(get_amp_cross)

    def get_b_cross(self):
        return self._c_b_cross.view()

    b_cross = property(get_b_cross)


cdef class PairPotDampDisp(PairPot):
    cdef long _c_nffatype
    cdef np.ndarray _c_c6_cross
    cdef np.ndarray _c_b_cross
    name = 'dampdisp'

    def __cinit__(self, np.ndarray[long, ndim=1] ffatype_ids not None,
                  np.ndarray[double, ndim=2] c6_cross not None,
                  np.ndarray[double, ndim=2] b_cross not None,
                  double rcut, Truncation tr=None,
                  np.ndarray[double, ndim=1] c6s=None,
                  np.ndarray[double, ndim=1] bs=None,
                  np.ndarray[double, ndim=1] vols=None):
        assert ffatype_ids.flags['C_CONTIGUOUS']
        assert c6_cross.flags['C_CONTIGUOUS']
        assert b_cross.flags['C_CONTIGUOUS']
        nffatype = c6_cross.shape[0]
        assert ffatype_ids.min() >= 0
        assert ffatype_ids.max() < nffatype
        assert c6_cross.shape[1] == nffatype
        assert b_cross.shape[0] == nffatype
        assert b_cross.shape[1] == nffatype
        if c6s is not None or vols is not None:
            assert c6s is not None
            assert vols is not None
            assert c6s.flags['C_CONTIGUOUS']
            assert vols.flags['C_CONTIGUOUS']
            assert c6s.shape[0] == nffatype
            assert bs.shape[0] == nffatype
            self._init_c6_cross(nffatype, c6_cross, c6s, vols)
        if bs is not None:
            assert bs.flags['C_CONTIGUOUS']
            self._init_b_cross(nffatype, b_cross, bs)
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_dampdisp_init(
            self._c_pair_pot, nffatype, <long*> ffatype_ids.data,
            <double*> c6_cross.data, <double*> b_cross.data,
        )
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_nffatype = nffatype
        self._c_c6_cross = c6_cross
        self._c_b_cross = b_cross

    def _init_c6_cross(self, nffatype, c6_cross, c6s, vols):
        for i0 in xrange(nffatype):
            for i1 in xrange(i0+1):
                if c6_cross[i0, i1] == 0.0 and vols[i0] != 0.0 and vols[i1] != 0.0:
                    ratio = vols[i0]/vols[i1]
                    c6_cross[i0, i1] = 2.0*c6s[i0]*c6s[i1]/(c6s[i0]/ratio+c6s[i1]*ratio)
                    c6_cross[i1, i0] = c6_cross[i0, i1]

    def _init_b_cross(self, nffatype, b_cross, bs):
        for i0 in xrange(nffatype):
            for i1 in xrange(i0+1):
                if b_cross[i0, i1] == 0.0 and bs[i0] != 0.0 and bs[i1] != 0.0:
                    b_cross[i0, i1] = 0.5*(bs[i0] + bs[i1])
                    b_cross[i1, i0] = b_cross[i0, i1]

    def log(self):
        if log.do_high:
            log.hline()
            log('ffatype_id0 ffatype_id1         C6          B')
            log.hline()
            for i0 in xrange(self._c_nffatype):
                for i1 in xrange(i0+1):
                    log('%11i %11i %s %s' % (i0, i1, log.c6(self._c_c6_cross[i0,i1]), log.invlength(self._c_b_cross[i0,i1])))

    def get_c6_cross(self):
        return self._c_c6_cross.view()

    c6_cross = property(get_c6_cross)

    def get_b_cross(self):
        return self._c_b_cross.view()

    b_cross = property(get_b_cross)


cdef class PairPotEI(PairPot):
    cdef np.ndarray _c_charges
    name = 'ei'

    def __cinit__(self, np.ndarray[double, ndim=1] charges, double alpha,
                  double rcut, Truncation tr=None):
        assert charges.flags['C_CONTIGUOUS']
        pair_pot.pair_pot_set_rcut(self._c_pair_pot, rcut)
        self.set_truncation(tr)
        pair_pot.pair_data_ei_init(self._c_pair_pot, <double*>charges.data, alpha)
        if not pair_pot.pair_pot_ready(self._c_pair_pot):
            raise MemoryError()
        self._c_charges = charges

    def log(self):
        if log.do_medium:
            log('  alpha:             %s' % log.invlength(self.alpha))
        if log.do_high:
            log.hline()
            log('   Atom     Charge')
            log.hline()
            for i in xrange(self._c_charges.shape[0]):
                log('%7i %s' % (i, log.charge(self._c_charges[i])))

    def get_charges(self):
        return self._c_charges.view()

    charges = property(get_charges)

    def get_alpha(self):
        return pair_pot.pair_data_ei_get_alpha(self._c_pair_pot)

    alpha = property(get_alpha)



#
# Ewald summation stuff
#


def compute_ewald_reci(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       Cell unitcell, double alpha,
                       np.ndarray[long, ndim=1] jmax, double kcut,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=1] work,
                       np.ndarray[double, ndim=2] vtens):
    cdef double *my_gpos
    cdef double *my_work
    cdef double *my_vtens

    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == pos.shape[0]
    assert unitcell.nvec == 3
    assert alpha > 0
    assert jmax.flags['C_CONTIGUOUS']
    assert jmax.shape[0] == 3

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
                                    unitcell._c_cell, alpha,
                                    <long*>jmax.data, kcut, my_gpos, my_work,
                                    my_vtens)


def compute_ewald_corr(np.ndarray[double, ndim=2] pos,
                       np.ndarray[double, ndim=1] charges,
                       Cell unitcell, double alpha,
                       np.ndarray[pair_pot.scaling_row_type, ndim=1] stab,
                       np.ndarray[double, ndim=2] gpos,
                       np.ndarray[double, ndim=2] vtens):
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
        <pair_pot.scaling_row_type*>stab.data, len(stab), my_gpos,
        my_vtens, len(pos)
    )


#
# Delta list
#


def dlist_forward(np.ndarray[double, ndim=2] pos,
                  Cell unitcell,
                  np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
    assert pos.flags['C_CONTIGUOUS']
    assert pos.shape[1] == 3
    assert deltas.flags['C_CONTIGUOUS']
    dlist.dlist_forward(<double*>pos.data, unitcell._c_cell,
                        <dlist.dlist_row_type*>deltas.data, ndelta)

def dlist_back(np.ndarray[double, ndim=2] gpos,
               np.ndarray[double, ndim=2] vtens,
               np.ndarray[dlist.dlist_row_type, ndim=1] deltas, long ndelta):
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
    assert deltas.flags['C_CONTIGUOUS']
    assert ictab.flags['C_CONTIGUOUS']
    iclist.iclist_forward(<dlist.dlist_row_type*>deltas.data,
                          <iclist.iclist_row_type*>ictab.data, nic)

def iclist_back(np.ndarray[dlist.dlist_row_type, ndim=1] deltas,
                np.ndarray[iclist.iclist_row_type, ndim=1] ictab, long nic):
    assert deltas.flags['C_CONTIGUOUS']
    assert ictab.flags['C_CONTIGUOUS']
    iclist.iclist_back(<dlist.dlist_row_type*>deltas.data,
                       <iclist.iclist_row_type*>ictab.data, nic)


#
# Valence list
#


def vlist_forward(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
                  np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    return vlist.vlist_forward(<iclist.iclist_row_type*>ictab.data,
                               <vlist.vlist_row_type*>vtab.data, nv)

def vlist_back(np.ndarray[iclist.iclist_row_type, ndim=1] ictab,
               np.ndarray[vlist.vlist_row_type, ndim=1] vtab, long nv):
    assert ictab.flags['C_CONTIGUOUS']
    assert vtab.flags['C_CONTIGUOUS']
    vlist.vlist_back(<iclist.iclist_row_type*>ictab.data,
                     <vlist.vlist_row_type*>vtab.data, nv)
