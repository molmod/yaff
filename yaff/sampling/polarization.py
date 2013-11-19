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
'''Polarizable forcefields'''

import numpy as np

from yaff.sampling import Hook

__all__ = ['RelaxDipoles']

class RelaxDipoles(Hook):
    def __init__(self, poltens, start=0, step=1):
        """
           **Arguments:**

           poltens
                Tensor that gives the atomic polarizabilities (3natom x 3 )

           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.poltens = poltens
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        #Implement determination of dipoles here.
        #Atomic positions and charges are available as iterative.ff.system.pos and
        #iterative.ff.system.charges
        #The dipoles can be set through iterative.ff.part_pair_eidip.pair_pot.dipoles

        #Check there is a pair_pot for dipoles present in the forcefield
        part_names = [part.name for part in iterative.ff.parts]
        assert 'pair_eidip' in part_names, "ff has to contain pair_eidip when using dipoles"
        #print iterative.ff.parts[0].name
        #print "Relaxing Dipoles"
        #print "Positions", iterative.ff.system.pos
        #print "Dipoles", iterative.ff.part_pair_eidip.pair_pot.dipoles
        #newdipoles = np.random.rand(np.shape( iterative.ff.part_pair_eidip.pair_pot.dipoles ) [0] , 3)
        #print "Next dipoles", newdipoles
        #iterative.ff.part_pair_eidip.pair_pot.dipoles = newdipoles
