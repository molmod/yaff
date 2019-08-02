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
'''Trial moves for Monte-Carlo simulations'''


from __future__ import division

import numpy as np

from yaff.log import log
from yaff.sampling.mcutils import *


__all__ = ['TrialInsertion','TrialDeletion','TrialRotation','TrialTranslation']


class Trial(object):
    """
    Base class for MC trial moves. Each trial move needs to be attached to
    an MC instance
    """
    def __init__(self, mc):
        self.mc = mc

    def __call__(self):
        """Perform a trial move, calculate the associated energy difference,
        decide whether it is accepted or not, and update the state of the
        MC simulation accordingly
        """
        e = self.calculate_energy()
        p, accepted = self.decide(e)
        if log.do_debug:
            log("MC %s: energy difference = %s acceptance probability = %6.2f %% accepted = %s"
                % (self.__class__.__name__, log.energy(e), p*100.0, accepted))
        return accepted

    def calculate_energy(self):
        # Subclasses implement their code here.
        raise NotImplementedError

    def decide(self, e):
        # Subclasses implement their code here.
        raise NotImplementedError


class TrialCartesian(Trial):
    """Base class for MC trial moves involving moving a randomly selected
    guest molecule in Cartesian space"""
    def calculate_energy(self):
        if self.mc.N==0:
            # No guests to move...
            e = 0.0
        else:
            iguest = np.random.randint(self.mc.N)
            # Select the guest-guest force field with correct number of guests
            ff = self.mc.get_ff(self.mc.N)
            # Reorder positions so the selected guest ends up last
            self.mc.reorder_guests(ff.system, iguest)
            # Original positions
            ff.update_pos(ff.system.pos)
            e = -ff.compute()+self.mc.eguest
            extpot = self.mc.external_potential
            self.oldpos = ff.system.pos[-self.mc.guest.natom:].copy()
            if extpot is not None:
                extpot.system.pos[-self.mc.guest.natom:] = self.oldpos
                extpot.update_pos(extpot.system.pos)
                e -= extpot.compute()-self.mc.eguest
            # Move the selected molecule in Cartesian space
            self.newpos = self.cartesian_move(self.oldpos)
            ff.system.pos[-self.mc.guest.natom:] = self.newpos
            ff.update_pos(ff.system.pos)
            e += ff.compute()-self.mc.eguest
            if extpot is not None:
                extpot.system.pos[-self.mc.guest.natom:] = self.newpos
                extpot.update_pos(extpot.system.pos)
                e += extpot.compute()-self.mc.eguest
        return e

    def decide(self, e):
        if self.mc.N==0: p = 0.0
        else: p = min(1.0, np.exp(-self.mc.beta*e))
        if np.random.rand()>p:
            accepted = False
            if self.mc.N>0:
                # Undo the Cartesian move
                ff = self.mc.get_ff(self.mc.N)
                ff.system.pos[-self.mc.guest.natom:] = self.oldpos
        else:
            accepted = True
        return p, accepted

    def cartesian_move(self, guestpos):
        # Subclasses implement their code here.
        raise NotImplementedError


class TrialRotation(TrialCartesian):
    """Random rotation of a randomly selected guest"""
    def cartesian_move(self, guestpos):
        # Move center to origin
        center = np.average(guestpos, axis=0)
        newpos = guestpos-center
        # Rotate randomly
        M = get_random_rotation_matrix()
        newpos = np.einsum('ib,ab->ia', newpos, M)
        # Move center to where it was originally
        return newpos+center


class TrialTranslation(TrialCartesian):
    """Random translation of a randomly selected guest"""
    def cartesian_move(self, guestpos):
        translation = self.mc.translation_stepsize*(np.random.rand(3)-0.5)
        return guestpos+translation


class TrialVolumechange(Trial):
    def calculate_energy(self):
        ff = self.mc.ff_full
        ff.system.pos[:] = self.mc.state.pos
        ff.update_pos(ff.system.pos)
        e -= ff.compute()
        self.oldrvecs = ff.system.cell.rvecs.copy()
        self.oldpos = ff.system.pos.copy()
        # New cell vectors
        # TODO
#        scale = 1.0 + 0.1*(np.random.rand()-0.5)
#        ff.system.cell.rvecs[:] *= scale
        raise NotImplementedError

    def decide(self, e):
        raise NotImplementedError


class TrialInsertion(Trial):
    """Insert a guest at a random position"""
    def calculate_energy(self):
        self.mc.N += 1
        # Select the guest-guest force field with correct number of guests
        ff = self.mc.get_ff(self.mc.N)
        # Set the positions of all other guests based on the current state
        if self.mc.N>1:
            ff.system.pos[:(self.mc.N-1)*self.mc.guest.natom] = self.mc.state.pos
        # Generate random guest configuration for the last (inserted) guest
        ff.system.pos[(self.mc.N-1)*self.mc.guest.natom:] = random_insertion(self.mc.guest)
        # Calculate the energy difference for guest-guest interactions
        ff.update_pos(ff.system.pos)
        e = ff.compute() - self.mc.eguest
        # Calculate the energy difference for guest-host interactions
        extpot = self.mc.external_potential
        if  extpot is not None:
            extpot.system.pos[-self.mc.guest.natom:] = ff.system.pos[-self.mc.guest.natom:]
            extpot.update_pos(extpot.system.pos)
            e += extpot.compute() - self.mc.eguest
        return e

    def decide(self, e):
        # Acceptance rule (Frenkel G.1.11), note that self.N is already updated
        p = min(1.0, self.mc.guest.cell.volume*self.mc.beta*self.mc.fugacity/self.mc.N*np.exp(-self.mc.beta*e))
        if np.random.rand()>p:
            # Reject MC move
            self.mc.N -= 1
            accepted = False
        else:
            # Accept MC move
            self.mc.state = self.mc.get_ff(self.mc.N).system
            self.mc.energy += e
            accepted = True
        return p, accepted


class TrialDeletion(Trial):
    def calculate_energy(self):
        """Delete a randomly selected guest"""
        if self.mc.N==0:
            e = 0.0
        else:
            iguest = np.random.randint(self.mc.N)
            # Select the guest-guest force field with correct number of guests
            ff = self.mc.get_ff(self.mc.N)
            # Reorder positions so the selected guest ends up last
            self.mc.reorder_guests(ff.system, iguest)
            # Calculate the energy difference for guest-guest interactions
            ff.update_pos(ff.system.pos)
            e = ff.compute() - self.mc.eguest
            # Calculate the energy difference for guest-host interactions
            extpot = self.mc.external_potential
            if extpot is not None:
                extpot.system.pos[-self.mc.guest.natom:] = ff.system.pos[-self.mc.guest.natom:]
                extpot.update_pos(extpot.system.pos)
                e += extpot.compute() - self.mc.eguest
        return e

    def decide(self, e):
        if self.mc.N==0:
            p = 0.0
        else:
            # Acceptance rule (based on Frenkel G.1.11)
            p = min(1.0, self.mc.N/(self.mc.guest.cell.volume*self.mc.beta*self.mc.fugacity)*np.exp(self.mc.beta*e))
        if np.random.rand()>p:
            # Reject MC move
            accepted = False
        else:
            # Accept MC move
            self.mc.N -= 1
            ff = self.mc.get_ff(self.mc.N)
            self.mc.state = ff.system
            self.mc.state.pos[:] = ff.system.pos[:self.mc.N*self.mc.guest.natom]
            self.mc.energy -= e
            accepted = True
        return p, accepted
