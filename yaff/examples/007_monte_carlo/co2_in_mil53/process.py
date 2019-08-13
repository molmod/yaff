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
import matplotlib.pyplot as plt
from glob import glob
import os

from molmod.units import bar

from yaff.conversion.raspa import read_raspa_loading

def make_plot():
    results = np.load('results.npy')
    plt.clf()
    plt.plot(results[:,0]/bar, results[:,1], marker='o', label='Yaff')
    # Read the RASPA results
    fns = sorted(glob(os.path.join('raspa','Output','System_0','*.data')))
    results_raspa = []
    for fn in fns:
        T, P, fugacity, N, Nerr = read_raspa_loading(fn)
        results_raspa.append([P,N])
    results_raspa = np.asarray(results_raspa)
    indexes = np.argsort(results_raspa[:,0])
    results_raspa = results_raspa[indexes]
    plt.plot(results_raspa[:,0]/bar, results_raspa[:,1], marker='.',
        label='RASPA', linestyle='--')
    plt.legend()
    plt.xlabel("P [bar]")
    plt.ylabel("Uptake [molecules/uc]")
    plt.savefig('results.png')

if __name__=='__main__':
    make_plot()
