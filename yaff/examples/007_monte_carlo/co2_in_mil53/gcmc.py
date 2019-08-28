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
'''GCMC simulation of rigid CO2 molecules inside the rigid MIL-53 framework'''


from __future__ import division
from __future__ import print_function

import numpy as np

from molmod.units import kelvin, bar, angstrom, kjmol, liter
from molmod.constants import avogadro

from yaff.pes.eos import PREOS
from yaff.sampling.mc import GCMC
from yaff.sampling.mcutils import MCScreenLog
from yaff.system import System
from yaff import log
log.set_level(log.medium)


def simulate():
    T = 298.0*kelvin
    # Setup the GCMC simulation
    fn_guest = 'CO2.chk'
    fn_host = 'MIL53.chk'
    fn_pars = ['pars.txt']
    host = System.from_file(fn_host).supercell(1,1,1)
    screenlog = MCScreenLog(step=10000)
    log.set_level(log.silent)
    gcmc = GCMC.from_files(fn_guest, fn_pars, host=host,
        rcut=12.0*angstrom, tr=None, tailcorrections=True, hooks=[screenlog],
        reci_ei='ewald_interaction', nguests=30)
    log.set_level(log.medium)
    # Description of allowed MC moves and their corresponding probabilities
    mc_moves =  {'insertion':1.0, 'deletion':1.0,
                 'translation':1.0, 'rotation':1.0}
    # Construct equation of state to link pressure, fugacity and chemical potential
    eos = PREOS.from_name('carbondioxide')
    # Loop over pressures to construct isotherm
    pressures = np.array([0.1,0.5,1.0,3.0,5.0,10.0])*bar
    uptake = np.zeros(pressures.shape)
    for iP, P in enumerate(pressures):
        fugacity = eos.calculate_fugacity(T,P)
        mu = eos.calculate_mu(T,P)
        # Set the external conditions
        gcmc.set_external_conditions(T, fugacity)
        # Run MC simulation
        gcmc.run(1000000, mc_moves=mc_moves)
        gcmc.current_configuration = None
        uptake[iP] = gcmc.Nmean
    np.save('results.npy', np.array([pressures,uptake]).T)

if __name__=='__main__':
    simulate()
