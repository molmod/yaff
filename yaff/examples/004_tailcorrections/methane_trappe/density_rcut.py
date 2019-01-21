#!/usr/bin/env python
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

import numpy as np
import h5py as h5
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22, 'legend.fontsize':16})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from yaff import log
from yaff.pes import NeighborList, Scalings, PairPotLJ, ForcePartPair, Switch3,\
    ForceField, ForcePart, ForcePartTailCorrection
from yaff.sampling import MTKBarostat, NHCThermostat, TBCombination, XYZWriter, VerletIntegrator,\
    HDF5Writer, VerletScreenLog
from yaff.test.common import get_system_fluidum_grid
from molmod.periodic import periodic
from molmod.units import meter, kilogram, kelvin, angstrom, kjmol, atm, femtosecond
from molmod.constants import boltzmann

'''
Simulate liquid methane using the TraPPE force field, which represents methane
molecules as single sites interacting through a Lennard-Jones potential.

The density at 110K is investigated as a function of the cut-off radius used
for the Lennard-Jones interactons with and without tail corrections.

TraPPE parameters can be found here:
    http://chem-siepmann.oit.umn.edu/siepmann/trappe/index.html
or in the paper
    Martin and Siepmann,
    Transferable Potentials for Phase Equilibria. 1. United-Atom Description of n-Alkanes
    J. Phys. Chem. B, 1998, 102 (14), pp 2569–2577
    http://dx.doi.org/10.1021/jp972543+
'''

def get_ff(system, rcut, tr=Switch3(2.0*angstrom), tailcorr=False):
    sigmas = 3.730*np.ones((system.natom,))*angstrom
    epsilons = 148.0*kelvin*boltzmann*np.ones((system.natom))
    nlist = NeighborList(system)
    scalings = Scalings(system)
    pair_lj = PairPotLJ(sigmas, epsilons, rcut, tr=tr)
    part_lj = ForcePartPair(system, nlist, scalings, pair_lj)
    parts = [part_lj]
    if tailcorr:
        part_tail = ForcePartTailCorrection(system, part_lj)
        parts.append(part_tail)
    ff = ForceField(system, parts, nlist)
    return ff


def run_md(system, rcut, tailcorr, tr=Switch3(2.0*angstrom), steps=2000, steps_eq=1000, fn_suffix=''):
    # Construct the force field
    ff = get_ff(system, rcut, tr=tr, tailcorr=tailcorr)
#    # Setup the integrator
    vsl = VerletScreenLog(step=100)
    nhc = NHCThermostat( T, start=0, timecon=100.0*femtosecond, chainlength=3)
    mtk = MTKBarostat(ff, T, P, start = 0, timecon=1000.0*femtosecond, anisotropic=False)
    hdf5 = HDF5Writer(h5.File('output%s.h5'%(fn_suffix), mode='w'), start=steps_eq, step=20)
    tbc = TBCombination(nhc, mtk)
    hooks = [vsl,tbc]
    # Production run
    verlet = VerletIntegrator(ff, 2.0*femtosecond, temp0=T,hooks=hooks+[hdf5])
    verlet.run(steps)
    # Read the volumes
    with h5.File('output%s.h5'%(fn_suffix),'a') as fh5:
        fh5.attrs.create('rcut',data=rcut)
        fh5.attrs.create('tailcorr',data=tailcorr)
        volume = fh5['trajectory/volume'][:]
    rho = np.sum(ff.system.masses)/np.mean(volume)
#    print("%20.2f %20.12f"%(rcut/angstrom,rho/kilogram*meter**3))
    return rho


def main(steps, steps_eq):
    log.set_level(log.medium)
    # Initial density
    rho0 = 420*kilogram/meter**3
    masses = np.ones((nmol,))*( periodic[6].mass + 4*periodic[1].mass )
    # Initial volume
    V0 = np.sum(masses)/rho0
    # Estimate intermolecular distance based on the volume per molecule
    l0 = (V0/nmol)**(1.0/3.0)
    # Setup the system
    system = get_system_fluidum_grid(nmol, ffatypes=['CH4'], l0=l0)
    system.masses = masses
    # Values of rcut for which a simulation will be performed
    rcuts = np.linspace(10.0,20.0,6)*angstrom
    results = []
    for ircut, rcut in enumerate(rcuts):
        row = [rcut]
        for tailcorr in [False,True]:
            fn_suffix = '_%04d_switch3_%03d'%(nmol,ircut)
            if tailcorr: fn_suffix += '_tailcorr'
            rho = run_md(system, rcut, tailcorr, steps=steps, steps_eq=steps_eq,fn_suffix=fn_suffix)
            row.append(rho)
        results.append(row)
    # Print summary
    print("%20s %20s %20s"%("rcut [AA]","rho [kg/meter**3]","rho with tail corrections [kg/meter**3]"))
    print("="*100)
    for rcut, rho0, rho1 in results:
        print("%20.2f %20.12f %20.12f"%(rcut/angstrom,rho0/kilogram*meter**3,rho1/kilogram*meter**3))


def process(nmol, suffix):
    fns = sorted(glob('output_%04d%s*.h5'%(nmol,suffix)))
    results = []
    for fn in fns:
        with h5.File(fn,'r') as fh5:
            rcut = fh5.attrs['rcut']
            tailcorr = fh5.attrs['tailcorr']
            mass = np.sum(fh5['system/masses'][:])
            volume = np.mean(fh5['trajectory/volume'][:])
            rho = mass/volume
        results.append([rcut,tailcorr,rho])
    results = np.asarray(results)
    print(results)
    results = np.load('results.npy')
    plt.clf()
    mask = results[:,1]==0
    plt.plot(results[mask,0]/angstrom,results[mask,2]/kilogram*meter**3,label='Without tail corrections', color='r', marker='o')
    mask = results[:,1]==1
    plt.plot(results[mask,0]/angstrom,results[mask,2]/kilogram*meter**3,label='With tail corrections', color='b', marker='o')
    plt.xlabel("$r_\mathrm{cut}\,[\mathrm{\AA}]$")
    plt.ylabel("$\\rho\,[\mathrm{kg}\,\mathrm{m}^{-3}]$")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('density_rcut.png')

if __name__=='__main__':
    nmol = 97       # Number of methane molecules
    T = 110.0*kelvin # simulation temperature
    P = 1.0*atm      # simulation pressure
    steps, steps_eq = ,10 # number of steps for production, number of steps for equilibration
    main(steps, steps_eq)
    process(nmol,'_switch3')
