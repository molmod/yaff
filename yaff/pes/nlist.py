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
'''Neighbor lists for pairwise (non-bonding) interactions

   Yaff works with half neighbor lists with relative vector information and with
   support for Verlet skin.

   Yaff supports only one neighbor list, which is used to evaluate all
   non-bonding interactions. The neighbor list is used by the ``ForcePartPair``
   objects. Each ``ForcePartPair`` object may have a different cutoff, of which
   the largest one determines the cutoff of the neighbor list. Unlike several
   other codes, Yaff uses one long neighbor list that contains all relevant atom
   pairs.

   The ``NeighborList`` object contains algorithms to detect whether a full rebuild
   of the neighbor list is required, or whether a recomputation of the distances
   and relative vectors is sufficient.
'''


import numpy as np

from yaff.log import log, timer
from yaff.pes.ext import nlist_status_init, nlist_status_finish, nlist_build, \
    nlist_recompute


__all__ = ['NeighborList']


neigh_dtype = [
    ('a', int), ('b', int), ('d', float),        # a & b are atom indexes, d is the distance
    ('dx', float), ('dy', float), ('dz', float), # relative vector (includes cell vectors of image cell)
    ('r0', int), ('r1', int), ('r2', int)        # position of image cell.
]


class NeighborList(object):
    '''Algorithms to keep track of all pair distances below a given rcut
    '''
    def __init__(self, system, skin=0):
        """
           **Arguments:**

           system
                A System instance.

           **Optional arguments:**

           skin
                A margin added to the rcut parameter. Only when atoms are
                displaced by half this distance, the neighbor list is rebuilt
                from scratch. In the other case, the distances of the known
                pairs are just recomputed. If set to zero, the default, the
                neighbor list is rebuilt at each update.

                A reasonable skin setting can drastically improve the
                performance of the neighbor list updates. For example, when
                ``rcut`` is ``10*angstrom``, a ``skin`` of ``2*angstrom`` is
                reasonable. If the skin is set too large, the updates will
                become very inefficient. Some tuning of ``rcut`` and ``skin``
                may be beneficial.
        """
        if skin < 0:
            raise ValueError('The skin parameter must be positive.')
        self.system = system
        self.skin = skin
        self.rcut = 0.0
        # the neighborlist:
        self.neighs = np.empty(10, dtype=neigh_dtype)
        self.nneigh = 0
        self.rmax = None
        # for skin algorithm:
        self._pos_old = None
        self.rebuild_next = False

    def request_rcut(self, rcut):
        """Make sure the internal rcut parameter is at least is high as rcut."""
        self.rcut = max(self.rcut, rcut)
        self.update_rmax()

    def update_rmax(self):
        """Recompute the ``rmax`` attribute.

           ``rmax`` determines the number of periodic images that are
           considered. when building the neighbor list. Along the a direction,
           images are taken from ``-rmax[0]`` to ``rmax[0]`` (inclusive). The
           range of images along the b and c direction are controlled by
           ``rmax[1]`` and ``rmax[2]``, respectively.

           Updating ``rmax`` may be necessary for two reasons: (i) the cutoff
           has changed, and (ii) the cell vectors have changed.
        """
        # determine the number of periodic images
        self.rmax = np.ceil((self.rcut+self.skin)/self.system.cell.rspacings-0.5).astype(int)
        if log.do_high:
            if len(self.rmax) == 1:
                log('rmax a       = %i' % tuple(self.rmax))
            elif len(self.rmax) == 2:
                log('rmax a,b     = %i,%i' % tuple(self.rmax))
            elif len(self.rmax) == 3:
                log('rmax a,b,c   = %i,%i,%i' % tuple(self.rmax))
        # Request a rebuild of the neighborlist because there is no simple way
        # to figure out whether an update is sufficient.
        self.rebuild_next = True

    def update(self):
        '''Rebuild or recompute the neighbor lists

           Based on the changes of the atomic positions or due to calls to
           ``update_rcut`` and ``update_rmax``, the neighbor lists will be
           rebuilt from scratch.

           The heavy computational work is done in low-level C routines. The
           neighbor lists array is reallocated if needed. The memory allocation
           is done in Python for convenience.
        '''
        with log.section('NLIST'), timer.section('Nlists'):
            assert self.rcut > 0

            if self._need_rebuild():
                # *rebuild* the entire neighborlist
                if self.system.cell.volume != 0:
                    if self.system.natom/self.system.cell.volume > 10:
                        raise ValueError('Atom density too high')
                # 1) make an initial status object for the neighbor list algorithm
                status = nlist_status_init(self.rmax)
                # 2) a loop of consecutive update/allocate calls
                last_start = 0
                while True:
                    done = nlist_build(
                        self.system.pos, self.rcut + self.skin, self.rmax,
                        self.system.cell, status, self.neighs[last_start:]
                    )
                    if done:
                        break
                    last_start = len(self.neighs)
                    new_neighs = np.empty((len(self.neighs)*3)/2, dtype=neigh_dtype)
                    new_neighs[:last_start] = self.neighs
                    self.neighs = new_neighs
                    del new_neighs
                # 3) get the number of neighbors in the list.
                self.nneigh = nlist_status_finish(status)
                if log.do_debug:
                    log('Rebuilt, size = %i' % self.nneigh)
                # 4) store the current state to check in future calls if we
                #    need to do a rebuild or a recompute.
                self._checkpoint()
                self.rebuild_next = False
            else:
                # just *recompute* the deltas and the distance in the
                # neighborlist
                nlist_recompute(self.system.pos, self._pos_old, self.system.cell, self.neighs[:self.nneigh])
                if log.do_debug:
                    log('Recomputed')

    def _checkpoint(self):
        '''Internal method called after a neighborlist rebuild.'''
        if self.skin > 0:
            # Only use the skin algorithm if this parameter is larger than zero.
            if self._pos_old is None:
                self._pos_old = self.system.pos.copy()
            else:
                self._pos_old[:] = self.system.pos

    def _need_rebuild(self):
        '''Internal method that determines if a rebuild is needed.'''
        if self.skin <= 0 or self._pos_old is None or self.rebuild_next:
            return True
        else:
            # Compute an upper bound for the maximum relative displacement.
            disp = np.sqrt(((self.system.pos - self._pos_old)**2).sum(axis=1).max())
            disp *= 2*(self.rmax.max()+1)
            if log.do_debug:
                log('Maximum relative displacement %s      Skin %s' % (log.length(disp), log.length(self.skin)))
            # Compare with skin parameter
            return disp >= self.skin


    def to_dictionary(self):
        """Transform current neighbor list into a dictionary.

           This is slow. Use this method for debugging only!
        """
        dictionary = {}
        for i in xrange(self.nneigh):
            key = (
                self.neighs[i]['a'], self.neighs[i]['b'], self.neighs[i]['r0'],
                self.neighs[i]['r1'], self.neighs[i]['r2']
            )
            value = np.array([
                self.neighs[i]['d'], self.neighs[i]['dx'],
                self.neighs[i]['dy'], self.neighs[i]['dz']
            ])
            dictionary[key] = value
        return dictionary


    def check(self):
        """Perform a slow internal consistency test.

           Use this for debugging only. It is assumed that self.rmax is set correctly.
        """
        # 0) Some initial tests
        assert (
            (self.neighs['a'][:self.nneigh] > self.neighs['b'][:self.nneigh]) |
            (self.neighs['r0'][:self.nneigh] != 0) |
            (self.neighs['r1'][:self.nneigh] != 0) |
            (self.neighs['r2'][:self.nneigh] != 0)
        ).all()
        # A) transform the current nlist into a set
        actual = self.to_dictionary()
        # B) Define loops of cell vectors
        if self.system.cell.nvec == 3:
            def rloops():
                for r2 in xrange(0, self.rmax[2]+1):
                    if r2 == 0:
                        r1_start = 0
                    else:
                        r1_start = -self.rmax[1]
                    for r1 in xrange(r1_start, self.rmax[1]+1):
                        if r2 == 0 and r1 == 0:
                            r0_start = 0
                        else:
                            r0_start = -self.rmax[0]
                        for r0 in xrange(r0_start, self.rmax[0]+1):
                            yield r0, r1, r2
        elif self.system.cell.nvec == 2:
            def rloops():
                for r1 in xrange(0, self.rmax[1]+1):
                    if r1 == 0:
                        r0_start = 0
                    else:
                        r0_start = -self.rmax[0]
                    for r0 in xrange(r0_start, self.rmax[0]+1):
                        yield r0, r1, 0

        elif self.system.cell.nvec == 1:
            def rloops():
                for r0 in xrange(0, self.rmax[0]+1):
                    yield r0, 0, 0
        else:
            def rloops():
                yield 0, 0, 0

        # C) Compute the nlists the slow way
        validation = {}
        nvec = self.system.cell.nvec
        for r0, r1, r2 in rloops():
            for a in xrange(self.system.natom):
                for b in xrange(a+1):
                    if r0!=0 or r1!=0 or r2!=0:
                        signs = [1, -1]
                    elif a > b:
                        signs = [1]
                    else:
                        continue
                    for sign in signs:
                        delta = self.system.pos[b] - self.system.pos[a]
                        self.system.cell.mic(delta)
                        delta *= sign
                        if nvec > 0:
                            self.system.cell.add_vec(delta, np.array([r0, r1, r2])[:nvec])
                        d = np.linalg.norm(delta)
                        if d < self.rcut + self.skin:
                            if sign == 1:
                                key = a, b, r0, r1, r2
                            else:
                                key = b, a, r0, r1, r2
                            value = np.array([d, delta[0], delta[1], delta[2]])
                            validation[key] = value

        # D) Compare
        wrong = False
        with log.section('NLIST'):
            for key0, value0 in validation.iteritems():
                value1 = actual.pop(key0, None)
                if value1 is None:
                    log('Missing:  ', key0)
                    log('  Validation %s %s %s %s' % (
                        log.length(value0[0]), log.length(value0[1]),
                        log.length(value0[2]), log.length(value0[3])
                    ))
                    wrong = True
                elif abs(value0 - value1).max() > 1e-10*log.length.conversion:
                    log('Different:', key0)
                    log('  Actual     %s %s %s %s' % (
                        log.length(value1[0]), log.length(value1[1]),
                        log.length(value1[2]), log.length(value1[3])
                    ))
                    log('  Validation %s %s %s %s' % (
                        log.length(value0[0]), log.length(value0[1]),
                        log.length(value0[2]), log.length(value0[3])
                    ))
                    log('  Difference %10.3e %10.3e %10.3e %10.3e' %
                        tuple((value0 - value1)/log.length.conversion)
                    )
                    log('  AbsMaxDiff %10.3e' %
                        (abs(value0 - value1).max()/log.length.conversion)
                    )
                    wrong = True
            for key1, value1 in actual.iteritems():
                log('Redundant:', key1)
                log('  Actual     %s %s %s %s' % (
                    log.length(value1[0]), log.length(value1[1]),
                    log.length(value1[2]), log.length(value1[3])
                ))
                wrong = True
        assert not wrong
