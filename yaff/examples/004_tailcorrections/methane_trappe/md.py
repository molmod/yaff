#!/usr/bin/env python

import numpy as np
import h5py as h5

from yaff import log
from yaff.pes import NeighborList, Scalings, PairPotLJ, ForcePartPair, Switch3,\
    ForceField, ForcePart, ForcePartTailCorrection
from yaff.sampling import MTKBarostat, NHCThermostat, TBCombination, XYZWriter, VerletIntegrator,\
    HDF5Writer, VerletScreenLog
from molmod.periodic import periodic
from molmod.units import meter, kilogram, kelvin, angstrom, kjmol, atm, femtosecond
from molmod.constants import boltzmann

from common import get_system, get_ff


'''
The density at 110K is investigated as a function of the cut-off radius used
for the Lennard-Jones interactons with and without tail corrections.
'''


def run_md(system, rcut, tailcorr, tr=Switch3(2.0*angstrom), steps=2000, steps_eq=1000, fn_suffix=''):
    # Construct the force field
    ff = get_ff(system, rcut, tr=tr, tailcorr=tailcorr)
    # Setup the integrator
    vsl = VerletScreenLog(step=100)
    nhc = NHCThermostat( T, start=0, timecon=100.0*femtosecond, chainlength=3)
    mtk = MTKBarostat(ff, T, P, start = 0, timecon=1000.0*femtosecond, anisotropic=False)
    hdf5 = HDF5Writer(h5.File('output%s.h5'%(fn_suffix), mode='w'), start=steps_eq, step=20)
    tbc = TBCombination(nhc, mtk)
    hooks = [vsl,tbc]
    # Production run
    verlet = VerletIntegrator(ff, 2.0*femtosecond, temp0=T,hooks=hooks+[hdf5])
    verlet.run(steps)
    # Write attributes
    with h5.File('output%s.h5'%(fn_suffix),'a') as fh5:
        fh5.attrs.create('rcut',data=rcut)
        fh5.attrs.create('tailcorr',data=tailcorr)
        volume = fh5['trajectory/volume'][:]
    rho = np.sum(ff.system.masses)/np.mean(volume)
    return rho


def main(steps, steps_eq):
    log.set_level(log.medium)
    # Initial density
    rho0 = 420*kilogram/meter**3
    system = get_system(nmol, rho0=rho0)
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


if __name__=='__main__':
    nmol = 97        # Number of methane molecules
    T = 110.0*kelvin # Simulation temperature
    P = 1.0*atm      # Simulation pressure
    # number of steps for production, number of steps for equilibration
    # to get proper results, you need quite a lot of steps (several 100 000)
    steps, steps_eq = 500000,100000
    main(steps, steps_eq)
