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

from yaff.log import log, timer
from yaff.sampling.mcutils import *

from molmod.units import kjmol


__all__ = ['Trial','TrialInsertion','TrialDeletion','TrialRotation',
    'TrialTranslation','TrialVolumechange']


class Trial(object):
    """
    Base class for MC trial moves. Each trial move needs to be attached to
    an MC instance
    """
    def __init__(self, mc):
        self.mc = mc

    def __call__(self):
        """Perform a trial move and calculate the associated energy difference,
        decide whether it is accepted or not, and update the state of the
        MC simulation accordingly
        """
        with timer.section("MC %s move" % self.log_name):
            e = self.compute()
            p = self.probability(e)
            if np.random.rand()>p:
                accepted = False
                self.reject()
            else:
                accepted = True
                self.mc.energy += e
                self.accept()
        if log.do_debug:
            log("MC %s: N = %d energy difference = %s acceptance probability = %6.2f %% accepted = %s"
                % (self.__class__.__name__, self.mc.N, log.energy(e), p*100.0, accepted))
        return accepted

    def insertion_energy(self, ff, sign=1):
        """Compute U(N+1)-U(N), assuming the inserted guest is positioned
        last."""
        assert sign in [-1,1]
        # Calculate the energy difference for guest-guest interactions
        ff.update_pos(ff.system.pos)
        e = ff.compute() - self.mc.eguest
        # Calculate the energy difference for guest-host interactions
        extpot = self.mc.external_potential
        if extpot is not None:
            extpot.system.pos[-self.mc.guest.natom:] = ff.system.pos[-self.mc.guest.natom:]
            extpot.update_pos(extpot.system.pos)
            e += extpot.compute() - self.mc.eguest
        # Energy difference for reciprocal Ewald (guest-guest and guest-host)
        if self.mc.ewald_reci is not None:
            if sign==1:
                cosfacs = self.mc.cosfacs_ins
                sinfacs = self.mc.sinfacs_ins
            else:
                cosfacs = self.mc.cosfacs_del
                sinfacs = self.mc.sinfacs_del
            # The insertion_energy method of ForcePartEwaldReciprocalInteraction already
            # takes the sign of the energy into account. We undo this by multiplying again
            # with `sign`, as we only correct our energy right at the end.
            e += sign*self.mc.ewald_reci.insertion_energy(ff.system.pos[-self.mc.guest.natom:],
                    ff.system.charges[-self.mc.guest.natom:],
                     cosfacs=cosfacs, sinfacs=sinfacs, sign=sign)
        return sign*e

    def compute(self):
        # Subclasses implement their code here.
        raise NotImplementedError

    def probability(self, e):
        # Subclasses implement their code here.
        raise NotImplementedError

    def accept(self):
        # Subclasses implement their code here.
        raise NotImplementedError

    def reject(self):
        # Subclasses implement their code here.
        raise NotImplementedError


class TrialCartesian(Trial):
    """Base class for MC trial moves involving moving a randomly selected
    guest molecule in Cartesian space"""
    def compute(self):
        if self.mc.N==0:
            # No guests to move; this should never be accepted, so we set the
            # energy to nan as it should not matter
            e = np.nan
        else:
            iguest = np.random.randint(self.mc.N)
            # Select the guest-guest force field with correct number of guests
            ff = self.mc.get_ff(self.mc.N)
            # Reorder positions so the selected guest ends up last
            self.mc.reorder_guests(ff.system, iguest)
            self.oldpos = ff.system.pos[-self.mc.guest.natom:].copy()
            # A translation/rotation can be considered as a deletion of a guest
            # at the original positions, followed by an insertion at the new
            # positions
            e = self.insertion_energy(ff, sign=-1)
            # Move the selected molecule in Cartesian space
            self.newpos = self.cartesian_move(self.oldpos)
            ff.system.pos[-self.mc.guest.natom:] = self.newpos
            e += self.insertion_energy(ff, sign=1)
        return e

    def probability(self, e):
        if self.mc.N==0:
            p = 0.0
        else:
            p = min(1.0, np.exp(-self.mc.beta*e))
        return p

    def accept(self):
        # Update the structure factors
        if self.mc.ewald_reci is not None and self.mc.N>0:
            self.mc.ewald_reci.cosfacs[:] += self.mc.cosfacs_ins
            self.mc.ewald_reci.sinfacs[:] += self.mc.sinfacs_ins

    def reject(self):
        if self.mc.N>0:
            # Undo the Cartesian move
            ff = self.mc.get_ff(self.mc.N)
            ff.system.pos[-self.mc.guest.natom:] = self.oldpos
            # Reset the Ewald structure factors
            if self.mc.ewald_reci is not None:
                self.mc.ewald_reci.cosfacs[:] += self.mc.cosfacs_del
                self.mc.ewald_reci.sinfacs[:] += self.mc.sinfacs_del

    def cartesian_move(self, guestpos):
        # Subclasses implement their code here.
        raise NotImplementedError


class TrialRotation(TrialCartesian):
    log_name = 'rot.'
    """Random rotation of a randomly selected guest"""
    def cartesian_move(self, guestpos):
        # Move center to origin
        center = np.mean(guestpos, axis=0)
        newpos = guestpos-center
        # Rotate randomly
        M = get_random_rotation_matrix()
        newpos = np.einsum('ib,ab->ia', newpos, M)
        # Move center to where it was originally
        return newpos + center


class TrialTranslation(TrialCartesian):
    log_name = 'trans.'
    """Random translation of a randomly selected guest"""
    def cartesian_move(self, guestpos):
        translation = self.mc.translation_stepsize*(np.random.rand(3)-0.5)
        return guestpos+translation


class TrialVolumechange(Trial):
    log_name = 'vol.'
    """Uniform rescaling of the cell vectors, keeping fractional coordinates"""
    def compute(self):
        assert self.mc.ewald_reci is None
        # Here there are no shortcuts; we need to compute the energy of the
        # entire system before and after the rescaling
        ff = self.mc.ff_full
        ff.system.pos[:] = self.mc.state.pos
        ff.update_pos(ff.system.pos)
        e = -ff.compute()
        self.oldrvecs = ff.system.cell.rvecs.copy()
        self.oldV = ff.system.cell.volume
        self.oldpos = ff.system.pos.copy()
        # Compute fractional coordinates
        frac = np.einsum('ab,ib->ia',ff.system.cell.gvecs,ff.system.pos)
        # Scaling factor based on requested largest volume change
        scale = np.power(1.0 + self.mc.volumechange_stepsize/ff.system.cell.volume,\
                 1.0/3.0) - 1.0
        assert scale>0.0
        assert scale<1.0
        self.newrvecs = self.oldrvecs*(1.0 + 2.0*scale*(np.random.rand()-0.5))
        ff.update_rvecs(self.newrvecs)
        self.newV = ff.system.cell.volume
        # Compute cartesian coordinates from fractional coordinates
        self.newpos = np.einsum('ab,ib->ia',ff.system.cell.rvecs,frac)
        ff.update_pos(self.newpos)
        e += ff.compute()
        return e

    def probability(self, e):
        p = np.exp(-self.mc.beta*(e+self.mc.P*(self.newV-self.oldV))+self.mc.N*np.log(self.newV/self.oldV))
        return min(1.0, p)

    def accept(self):
        self.mc.state.pos[:] = self.newpos
        self.mc.state.cell.update_rvecs(self.newrvecs)

    def reject(self):
        self.mc.state.pos[:] = self.oldpos
        self.mc.state.cell.update_rvecs(self.oldrvecs)


class TrialInsertion(Trial):
    log_name = 'ins.'
    """Insert a guest at a random position"""
    def compute(self):
        # e contains U(N+1) - U(N)
        self.mc.N += 1
        # Select the guest-guest force field with correct number of guests
        ff = self.mc.get_ff(self.mc.N)
        # Set the positions of all other guests based on the current state
        if self.mc.N>1:
            ff.system.pos[:(self.mc.N-1)*self.mc.guest.natom] = self.mc.state.pos
        # Generate random guest configuration for the last (inserted) guest
        ff.system.pos[(self.mc.N-1)*self.mc.guest.natom:] = random_insertion(self.mc.guest)
        return self.insertion_energy(ff)

    def probability(self, e):
        # Acceptance rule (Frenkel G.1.11), note that self.N is already updated
        return min(1.0, self.mc.guest.cell.volume*self.mc.beta*self.mc.fugacity/self.mc.N*np.exp(-self.mc.beta*e))

    def accept(self):
        # Set state to the system including the inserted guest
        self.mc.state = self.mc.get_ff(self.mc.N).system
        # Update the Ewald structure factors
        if self.mc.ewald_reci is not None:
            self.mc.ewald_reci.cosfacs[:] += self.mc.cosfacs_ins
            self.mc.ewald_reci.sinfacs[:] += self.mc.sinfacs_ins
            self.mc.cosfacs_ins[:] = 0.0
            self.mc.sinfacs_ins[:] = 0.0

    def reject(self):
        # Reset number of guest molecules
        self.mc.N -= 1


class TrialDeletion(Trial):
    log_name = 'del.'
    def compute(self):
        """Delete a randomly selected guest"""
        if self.mc.N==0:
            return 0.0
        else:
            # e contains U(N) - U(N-1)
            iguest = np.random.randint(self.mc.N)
            # Select the guest-guest force field with correct number of guests
            ff = self.mc.get_ff(self.mc.N)
            # Reorder positions so the selected guest ends up last
            self.mc.reorder_guests(ff.system, iguest)
            return self.insertion_energy(ff, sign=-1)

    def probability(self, e):
        if self.mc.N==0:
            p = 0.0
        else:
            # Acceptance rule (based on Frenkel G.1.11, note that e=U(N-1)-U(N))
            p = min(1.0, self.mc.N/(self.mc.guest.cell.volume*self.mc.beta*self.mc.fugacity)*np.exp(-self.mc.beta*e))
        return p

    def accept(self):
        # Update number of guests
        self.mc.N -= 1
        # Set the state to the system without the deleted guest
        ff = self.mc.get_ff(self.mc.N)
        ff.system.pos[:] = self.mc.get_ff(self.mc.N+1).system.pos[:self.mc.N*self.mc.guest.natom]
        self.mc.state = ff.system

    def reject(self):
        # Reset the Ewald structure factors
        if self.mc.ewald_reci is not None and self.mc.N>0:
            self.mc.ewald_reci.cosfacs[:] += self.mc.cosfacs_del
            self.mc.ewald_reci.sinfacs[:] += self.mc.sinfacs_del
