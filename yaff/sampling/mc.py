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
'''Monte-Carlo routines

   An important difference between molecular dynamics and Monte-Carlo (MC)
   simulations, is that in MC simulations one often only needs to know the
   energy difference between two configurations where a very small number of
   particles are displaced (for instance translating, rotating, inserting, or
   deleting a single molecule). For pairwise-additive force fields it is
   computationally very inefficient to compute all interactions in such a case:
   only the interactions involving the displaced particles should be computed.

   This can be achieved by using the 'n_frame' keyword when generating force
   fields, which ensures that interactions between the first 'n_frame' atoms
   are not calculated.

   TODO::
        * Extension to mixtures of guest molecules
        * Complete NPT MC simulator
        * Variable cell shape simulations?
        * Hybrid MD/MC
        * Tabulating the external potential
'''


from __future__ import division

import numpy as np

from molmod import boltzmann, femtosecond, angstrom, kelvin, bar

from yaff.log import log, timer
from yaff.pes.ff import ForceField, \
    ForcePartEwaldReciprocalInteraction
from yaff.pes.ext import Cell
from yaff.sampling.mcutils import *
from yaff.sampling import mctrials
from yaff.sampling.iterative import AttributeStateItem
from yaff.system import System


__all__ = [
    'MC', 'CanonicalMC', 'NPTMC', 'GCMC'
]


class MC(object):
    """Base class for Monte-Carlo simulations"""
    allowed_trials = []
    default_trials = {}
    default_state = []

    def __init__(self, state):
        """
            **Optional arguments:**
      
            state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                MC algorithm.
        """
        self.counter = 0
        if state is None:
            self.state_list = [state_item.copy() for state_item in self.default_state]
        else:
            self.state_list = [state_item.copy() for state_item in self.default_state]
            self.state_list += state
        self.state = dict((item.key, item) for item in self.state_list)
        self.call_hooks()

    def call_hooks(self):
        # Initialize hooks
        with timer.section('%s hooks' % self.log_name):
            state_updated = False
            for hook in self.hooks:
                if hook.expects_call(self.counter):
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    hook(self)

    def run(self, nsteps, mc_moves=None, initial=None, einit=0,
                translation_stepsize=1.0*angstrom,
                volumechange_stepsize=10.0*angstrom**3):
        """
           Perform Monte-Carlo steps

           **Arguments:**

           nsteps
                Number of Monte-Carlo steps

           **Optional Arguments:**

           mc_moves
                Dictionary containing relative probabilities of the different
                Monte-Carlo moves. It is not required that the probabilities
                sum to 1, they are normalized automatically. Example

           initial
                System instance describing the initial configuration of guest
                molecules

           einit
                The energy of the initial state

           translation_stepsize
                The maximal magnitude of a TrialTranslation

           volumechange_stepsize
                The maximal magnitude of a TrialVolumechange
        """
        if log.do_warning:
            log.warn("Currently, Yaff does not consider interactions of a guest molecule "
                     "with its periodic images in MC simulations. Make sure that you choose a system size "
                     "that is large compared to the guest dimensions, so it is indeed "
                     "acceptable to neglect these interactions.")
        with log.section(self.log_name), timer.section(self.log_name):
            # Initialization
            self.translation_stepsize = translation_stepsize
            self.volumechange_stepsize = volumechange_stepsize
            if initial is not None:
                self.N = initial.natom//self.guest.natom
                assert self.guest.natom*self.N==initial.natom, ("Initial configuration does not contain correct number of atoms")
                self.current_configuration = initial
                self.get_ff(self.N).system.pos[:] = initial.pos
                if self.ewald_reci is not None:
                    self.ewald_reci.compute_structurefactors(
                        initial.pos,
                        initial.charges,
                        self.ewald_reci.cosfacs, self.ewald_reci.sinfacs)
            else:
                self.current_configuration = self.get_ff(self.N).system
            self.energy = einit
            if not self.conditions_set:
                raise ValueError("External conditions have not been set!")
            # Normalized probabilities and accompanying methods specifying the trial MC moves
            # Trial moves are sorted alphabetically
            if mc_moves is None: mc_moves = self.default_trials
            trials, probabilities = [], []
            for t in sorted(mc_moves.keys()):
                if not t in self.allowed_trials:
                    raise ValueError("Trial move %s not allowed!"%t)
                trial = getattr(mctrials,"Trial"+t.capitalize(),None)
                if trial is None:
                    raise NotImplementedError("The requested trial move %s is not implemented"%(t))
                # Trials is a list containing instances of Trial classes from the mctrials module
                trials.append(trial(self))
                probabilities.append(mc_moves[t])
            probabilities = np.asarray(probabilities)
            probabilities /= np.sum(probabilities)
            assert np.all(probabilities>=0.0), "Negative probabilities are not allowed!"
            # Take the cumulative sum, makes it a bit easier to determine which MC move is selected
            probabilities = np.cumsum(probabilities)
            # Array to keep track of accepted (1st column) and tried (2nd column)
            # moves, with rows corresponding to different possible moves
            acceptance = np.zeros((len(trials),2), dtype=int)
            # Start performing MC moves
            self.Nmean = self.N
            self.emean = self.energy
            self.Vmean = self.current_configuration.cell.volume
            self.counter = 0
            for istep in range(nsteps):
                switch = np.random.rand()
                # Select one of the possible MC moves
                imove = np.where(switch<probabilities)[0][0]
                # Call the corresponding method
                accepted = trials[imove]()
                # Update records with accepted and tried MC moves
                acceptance[imove,1] += 1
                if accepted: acceptance[imove,0] += 1
                self.counter += 1
                self.Nmean += (self.N-self.Nmean)/self.counter
                self.emean += (self.energy-self.emean)/self.counter
                self.Vmean += (self.current_configuration.cell.volume-self.Vmean)/self.counter
                self.call_hooks()
            return acceptance

    def reorder_guests(self, system, iguest):
        """Reorder guests so the one with index iguest becomes the last one"""
        idx0 = np.concatenate((np.arange(iguest*self.guest.natom,(iguest+1)*self.guest.natom),
                               np.arange((self.N-1)*self.guest.natom,self.N*self.guest.natom)))
        idx1 = np.concatenate((np.arange((self.N-1)*self.guest.natom,self.N*self.guest.natom),
                               np.arange(iguest*self.guest.natom,(iguest+1)*self.guest.natom)))
        system.pos[idx1] = system.pos[idx0]

    def initialize_structure_factors(self, ff):
        """Efficient treatment of reciprocal Ewald summation"""
        self.ewald_reci = None
        self.cosfacs_ins, self.sinfacs_ins, self.cosfacs_del, self.cosfacs_ins = None, None, None, None
        for part in ff.parts:
            if isinstance(part, ForcePartEwaldReciprocalInteraction):
                self.ewald_reci = part
                self.cosfacs_ins = np.zeros(self.ewald_reci.cosfacs.shape)
                self.sinfacs_ins = np.zeros(self.ewald_reci.cosfacs.shape)
                self.cosfacs_del = np.zeros(self.ewald_reci.cosfacs.shape)
                self.sinfacs_del = np.zeros(self.ewald_reci.cosfacs.shape)
        if self.ewald_reci is not None and self.external_potential is not None:
            nfw = self.external_potential.system.natom-self.guest.natom
            self.ewald_reci.compute_structurefactors(
                    self.external_potential.system.pos[:nfw],
                    self.external_potential.system.charges[:nfw],
                    self.ewald_reci.cosfacs, self.ewald_reci.sinfacs)


class FixedNMC(MC):
    """Base class for Monte-Carlo simulations with fixed number of particles N"""
    allowed_trials = []
    default_trials = {}

    def __init__(self, guest, ff, external_potential=None, eguest=0.0,
        hooks=[], state=None):
        # Initialization
        self.guest = guest
        if guest.cell.nvec==0:
            raise TypeError('The system must be periodic for Canonical MC simulations')
        self.conditions_set = False
        self.ff = ff
        self.external_potential = external_potential
        self.eguest = eguest
        self.hooks = hooks
        self.initialize_structure_factors(self.ff)
        # The System describing the current configuration of guest molecules
        self.current_configuration = self.ff.system
        self.N = self.ff.system.natom//self.guest.natom
        assert self.N*self.guest.natom==self.ff.system.natom
        MC.__init__(self, state)

    def get_ff(self, nguests):
        assert nguests == self.N
        return self.ff


class CanonicalMC(FixedNMC):
    """Canonical Monte-Carlo simulations for rigid molecules (referred to
       as guests), optionally subjected to an external potential (for instance
       by adsorption in a rigid framework).
    """
    allowed_trials = ['translation','rotation']
    default_trials = {'translation': 0.5, 'rotation':0.5}
    log_name = 'NVTMC'
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('energy'),
        AttributeStateItem('emean'),
    ]

    def __init__(self, guest, ff, external_potential=None, eguest=0.0, 
        hooks=[], state=None):
        """
           **Arguments:**

           guest
                A System instance representing the species that is studied

           ff
                A ForceField instance describing guest-guest interactions

           **Optional Arguments:**

           external_potential
                A ForceField instance describing the potential that a single
                guest molecule experiences. It is mandatory that this single
                guest molecule is listed last in the corresponding System.
                This can for instance be used to describe adsorption in a
                rigid framework, where the rigid framework can be seen as an
                external potential for the guest molecules. A ForceField
                describing the framework-guest interactions can be easily
                constructed by making use of the n_frame keyword when
                generating a ForceField

            eguest
                The intramolecular energy of one guest in the gas phase;
                currently, guest molecules are assumed to be rigid so this
                will be constant throughout the simulation

            hooks
                A list of MCHooks

            state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                MC algorithm.
        """
        self.set_conditions(T)
        super(CanonicalMC, self).__init__(guest, ff,
            external_potential=external_potential, eguest=0.0, hooks=hooks,
            state=state)

    def set_external_conditions(self, T):
        # External conditions
        assert T>0.0
        self.T = T
        self.beta = 1.0/boltzmann/self.T
        self.conditions_set = True

    def log_header(self):
        return "%10s %10s" % ("E","<E>")

    def log(self):
        return '%s %s' % (log.energy(self.energy), log.energy(self.emean))


class NPTMC(FixedNMC):
    """Monte-Carlo simulations for rigid molecules (referred to
       as guests) in the isobaric-isothermal ensemble. For now, cell shapes
       are fixed
    """
    allowed_trials = ['translation','rotation','volumechange']
    default_trials = {'translation': 0.4, 'rotation':0.4, 'volumechange':0.2}
    log_name = 'NPTMC'
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('energy'),
        AttributeStateItem('emean'),
        MCVolumeStateItem(),
        AttributeStateItem('Vmean'),
    ]


    def __init__(self, guest, ff, ff_full, eguest=0.0, hooks=[], state=None):
        """
           **Arguments:**

           guest
                A System instance representing the species that is studied

           ff
                A ForceField instance describing interactions of the last guest
                with all other guests

           ff_full
                A ForceField instance describing all guest-guest interactions

           **Optional Arguments:**

           eguest
                The intramolecular energy of one guest in the gas phase;
                currently, guest molecules are assumed to be rigid so this
                will be constant throughout the simulation

           hooks
                A list of MCHooks

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                MC algorithm.
        """
        self.ff_full = ff_full
        super(NPTMC, self).__init__(guest, ff,
            external_potential=None, eguest=eguest, hooks=hooks, state=state)
        if self.ewald_reci is not None:
            raise ValueError("NPTMC simulations can not be performed when "
                             "ForcePartEwaldReciprocalInteraction is present")

    def set_external_conditions(self, T, P):
        """

           **Optional Arguments:**

           T
                Temperature
           P
                Pressure
        """
        assert T>0.0
        self.T = T
        self.beta = 1.0/boltzmann/self.T
        self.P = P
        self.conditions_set = True

    def log_header(self):
        return "%10s %10s %10s %10s" % ("V","<V>","E","<E>")


    def log(self):
        return '%s %s %s %s' % ( log.volume(self.current_configuration.cell.volume),
                log.volume(self.Vmean), log.energy(self.energy), log.energy(self.emean))


class GCMC(MC):
    """Grand Canonical Monte-Carlo simulations for rigid molecules (referred to
       as guests), optionally subjected to an external potential (for instance
       by adsorpion in a rigid framework).
    """
    allowed_trials = ['insertion','deletion','translation','rotation']
    default_trials = {'insertion':0.25, 'deletion':0.25,
                      'translation': 0.25, 'rotation':0.25}
    log_name = 'GCMC'
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('energy'),
        AttributeStateItem('emean'),
        AttributeStateItem('N'),
        AttributeStateItem('Nmean'),
    ]

    def __init__(self, guest, ff_generator, external_potential=None, eguest=0.0,
                 hooks=[], nguests=10, state=None):
        """
           **Arguments:**

           guest
                A System instance representing the species that is studied

           ff_generator
                A method that return a ForceField instance describing
                guest-guest interactions. As the number of guests varies during
                a simulation, we need to be able to generate force fields for
                different numbers of guests.

           **Optional Arguments:**

           external_potential
                A ForceField instance describing the potential that a single
                guest molecule experiences. It is mandatory that this single
                guest molecule is listed last in the corresponding System.
                This can for instance be used to describe adsorption in a
                rigid framework, where the rigid framework can be seen as an
                external potential for the guest molecules. A ForceField
                describing the framework-guest interactions can be easily
                constructed by making use of the n_frame keyword when
                generating a ForceField

            eguest
                The intramolecular energy of one guest in the gas phase;
                currently, guest molecules are assumed to be rigid so this
                will be constant throughout the simulation

            hooks
                A list of GCMCHooks

            nguests
                An initial estimate for the number of adsorbed guests. This is
                only used to initialize the force fields before the start of
                the simulation. If more than nguests are adsorbed, additional
                force fields will be generated on the fly.

            state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                MC algorithm.
        """
        # Initialization
        if guest.cell.nvec==0:
            raise TypeError('The system must be periodic for GCMC simulations')
        self.guest = guest
        self.ff_generator = ff_generator
        self.conditions_set = False
        self.eguest = eguest
        self.hooks = hooks
        # Generate some guest-guest force fields;
        self._ffs = []
        self._generate_ffs(nguests)
        self.external_potential = external_potential
        self.initialize_structure_factors(self.get_ff(1))
        # The System describing the current configuration of guest molecules
        self.current_configuration = None
        self.N = 0
        self.Nmean = 0.0
        self.energy = 0.0
        self.emean = 0.0
        MC.__init__(self, state)

    def set_external_conditions(self, T, fugacity):
        """
           **Arguments:**

           T
                Temperature

           fugacity
                Fugacity. If the species behaves more or less like an ideal gas,
                this is equal to the pressure. Otherwise, the fugacity can be
                obtained from the pressure using an equation of state from
                yaff.pes.eos
        """
        # External conditions
        assert T>0.0
        self.T = T
        self.beta = 1.0/boltzmann/self.T
        assert fugacity>=0.0
        self.fugacity = fugacity
        self.conditions_set = True
        if log.do_medium:
            with log.section(self.log_name):
                # log.pressure does not exist yet, what a pity...
                log("GCMC simulation with T = %s and fugacity = %12.6f bar"%(
                    log.temperature(self.T), self.fugacity/bar))

    def log_header(self):
        return "%10s %10s %10s %10s" % ("N","<N>","E","<E>")

    def log(self):
        return '%10d %10.6f %s %s' % ( self.N, self.Nmean,
                log.energy(self.energy), log.energy(self.emean))

    def _generate_ffs(self, nguests):
        for iguest in range(len(self._ffs),nguests):
            if len(self._ffs)==0:
                # The very first force field, no guests
                system = System.create_empty()
                system.cell = Cell(self.guest.cell.rvecs)
            elif len(self._ffs)==1:
                # The first real force field, a single guest
                system = self.guest
            else:
                # Take the system of the lastly generated force field (N-1) guests
                # and add an additional guest
                system = self._ffs[-1].system.merge(self.guest)
            self._ffs.append(self.ff_generator(system, self.guest))

    def get_ff(self, nguests):
        if nguests>=len(self._ffs):
            self._generate_ffs(nguests+1)
        return self._ffs[nguests]

    @classmethod
    def from_files(cls, guest, parameters, **kwargs):
        """Automated setup of GCMC simulation

           **Arguments:**

           guest
                Two types are accepted: (i) the filename of a system file
                describing one guest molecule, (ii) a System instance of
                one guest molecule

           parameters
                Force-field parameters describing guest-guest and optionally
                host-guest interaction.
                Three types are accepted: (i) the filename of the parameter
                file, which is a text file that adheres to YAFF parameter
                format, (ii) a list of such filenames, or (iii) an instance of
                the Parameters class.

           **Optional arguments:**

           hooks
                A list of MCHooks

           host
                Two types are accepted: (i) the filename of a system file
                describing the host system, (ii) a System instance of the host

           All other keyword arguments are passed to the ForceField constructor
           See the constructor of the :class:`yaff.pes.generator.FFArgs` class
           for the available optional arguments.

        """
        # Load the guest System
        if isinstance(guest, str):
            guest = System.from_file(guest)
        assert isinstance(guest, System)
        # We want to control nlow and nhigh here ourselves, so remove it from the
        # optional arguments if the user provided it.
        kwargs.pop('nlow', None)
        kwargs.pop('nhigh', None)
        # Rough guess for number of adsorbed guests
        nguests = kwargs.pop('nguests', 10)
        # Load the host if it is present as a keyword
        host = kwargs.pop('host', None)
        # Extract the hooks
        hooks = kwargs.pop('hooks', [])
        # Efficient treatment of reciprocal ewald contribution
        if not 'reci_ei' in kwargs.keys():
            kwargs['reci_ei'] = 'ewald_interaction'
        if host is not None:
            if isinstance(host, str):
                host = System.from_file(host)
            assert isinstance(host, System)
            # If the guest molecule is currently an isolated molecule, than put
            # it in the same periodic box as the host
            if guest.cell is None or guest.cell.nvec==0:
                guest.cell = Cell(host.cell.rvecs)
            # Construct a complex of host and one guest and the corresponding
            # force field excluding host-host interactions
            hostguest = host.merge(guest)
            external_potential = ForceField.generate(hostguest, parameters,
                 nlow=host.natom, nhigh=host.natom, **kwargs)
        else:
            external_potential = None
#        # Compare the energy of the guest, once isolated, once in a periodic box
#        guest_isolated = guest.subsystem(np.arange(guest.natom))
#        guest_isolated.cell = Cell(np.zeros((0,3)))
#        optional_arguments = {}
#        for key in kwargs.keys():
#            if key=='reci_ei': continue
#            optional_arguments[key] = kwargs[key]
#        ff_guest_isolated = ForceField.generate(guest_isolated, parameters, **optional_arguments)
#        e_isolated = ff_guest_isolated.compute()
#        guest_periodic = guest.subsystem(np.arange(guest.natom))
#        ff_guest_periodic = ForceField.generate(guest_periodic, parameters, **optional_arguments)
#        e_periodic = ff_guest_periodic.compute()
#        if np.abs(e_isolated-e_periodic)>1e-4:
#            if log.do_warning:
#                log.warn("An interaction energy of %s of the guest with its periodic "
#                         "images was detected. The interaction of a guest with its periodic "
#                         "images will however NOT be taken into account in this simulation. "
#                         "If the energy difference is large compared to k_bT, you should "
#                         "consider using a supercell." % (log.energy(e_isolated-e_periodic)))
        # By making use of nlow=nhigh, we automatically discard intramolecular energies
        eguest = 0.0
        # Generator of guest-guest force fields, excluding interactions
        # between the first N-1 guests
        def ff_generator(system, guest):
            return ForceField.generate(system, parameters, nlow=max(0,system.natom-guest.natom), nhigh=max(0,system.natom-guest.natom), **kwargs)
        return cls(guest, ff_generator, external_potential=external_potential,
             eguest=eguest, hooks=hooks, nguests=nguests)
