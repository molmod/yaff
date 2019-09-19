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


from __future__ import division

import numpy as np
import pkg_resources

from yaff import *
from molmod.units import angstrom, bar, kelvin, kcalmol
from molmod.constants import boltzmann


def check_insertion_energy(gcmc, fn_host, fn_pars, fn_guest):
    trial = TrialInsertion(gcmc)
    e = trial.compute()
    system0 = System.from_file(fn_host).merge(trial.mc.get_ff(trial.mc.N).system)
    ff0 = ForceField.generate(system0, fn_pars)
    system1 = system0.subsystem(np.arange(system0.natom-gcmc.guest.natom))
    ff1 = ForceField.generate(system1, fn_pars)
    system2 = system0.subsystem(np.arange(system0.natom-gcmc.guest.natom,system0.natom))
    ff2 = ForceField.generate(system2, fn_pars)
    eref = ff0.compute() - ff1.compute() - ff2.compute()
    assert np.abs(e-eref)<1e-10
    trial.reject()


def check_deletion_energy(gcmc, fn_host, fn_pars, fn_guest):
    trial = TrialDeletion(gcmc)
    e = trial.compute()
    system0 = System.from_file(fn_host).merge(trial.mc.get_ff(trial.mc.N).system)
    ff0 = ForceField.generate(system0, fn_pars)
    system1 = system0.subsystem(np.arange(system0.natom-gcmc.guest.natom))
    ff1 = ForceField.generate(system1, fn_pars)
    system2 = system0.subsystem(np.arange(system0.natom-gcmc.guest.natom,system0.natom))
    ff2 = ForceField.generate(system2, fn_pars)
    eref = -(ff0.compute() - ff1.compute() - ff2.compute())
    assert np.abs(e-eref)<1e-10
    trial.reject()


def check_translation_energy(gcmc, fn_host, fn_pars, fn_guest):
    trial = TrialTranslation(gcmc)
    e = trial.compute()
    system0 = System.from_file(fn_host).merge(trial.mc.get_ff(trial.mc.N).system)
    ff0 = ForceField.generate(system0, fn_pars)
    e0 = ff0.compute()
    system0.pos[-gcmc.guest.natom:] = trial.oldpos
    ff0.update_pos(system0.pos)
    e1 = ff0.compute()
    eref = e0-e1
    assert np.abs(e-eref)<1e-10
    trial.reject()


def check_rotation_energy(gcmc, fn_host, fn_pars, fn_guest):
    trial = TrialRotation(gcmc)
    e = trial.compute()
    # Host and guests after rotation
    system0 = System.from_file(fn_host).merge(trial.mc.get_ff(trial.mc.N).system)
    ff0 = ForceField.generate(system0, fn_pars)
    # One single guest, in a periodic box
    system2 = system0.subsystem(np.arange(system0.natom-gcmc.guest.natom,system0.natom))
    ff2 = ForceField.generate(system2, fn_pars)
    e0 = ff0.compute() - ff2.compute()
    # Host and guests before rotation
    system0.pos[-gcmc.guest.natom:] = trial.oldpos
    ff0.update_pos(system0.pos)
    ff2.update_pos(trial.oldpos)
    e1 = ff0.compute()-ff2.compute()
    eref = e0-e1
    assert np.abs(e-eref)<1e-10
    trial.reject()


def test_trials_xylene_in_cau13():
    fn_host = pkg_resources.resource_filename(__name__, '../../data/test/CAU_13.chk')
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_CAU-13_xylene.txt')
    fn_guest = pkg_resources.resource_filename(__name__, '../../data/test/xylene.chk')
    gcmc = GCMC.from_files(fn_guest, fn_pars, host=fn_host)
    # Insert some guests randomly
    init = System.from_file(fn_guest)
    for i in range(3):
        pos = random_insertion(gcmc.guest)
        guest = System.from_file(fn_guest)
        guest.pos[:] = pos
        init = init.merge(guest)
    gcmc.set_external_conditions(300*kelvin, 100*bar)
    gcmc.run(0, initial=init, close_contact=0.0)
    # Try out different moves
    check_insertion_energy(gcmc, fn_host, fn_pars, fn_guest)
    check_deletion_energy(gcmc, fn_host, fn_pars, fn_guest)
    check_translation_energy(gcmc, fn_host, fn_pars, fn_guest)
    check_rotation_energy(gcmc, fn_host, fn_pars, fn_guest)
