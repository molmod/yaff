#!/usr/bin/env python

import numpy as np
import h5py as h5

from yaff import log
from yaff.pes import NeighborList, Scalings, PairPotLJ, ForcePartPair, Switch3,\
    ForceField, ForcePart, ForcePartTailCorrection
from yaff.test.common import get_system_fluidum_grid
from molmod.periodic import periodic
from molmod.units import meter, kilogram, angstrom, kelvin
from molmod.constants import boltzmann


'''
Simulate liquid methane using the TraPPE force field, which represents methane
molecules as single sites interacting through a Lennard-Jones potential.

TraPPE parameters can be found here:
    http://chem-siepmann.oit.umn.edu/siepmann/trappe/index.html
or in the paper
    Martin and Siepmann,
    Transferable Potentials for Phase Equilibria. 1. United-Atom Description of n-Alkanes
    J. Phys. Chem. B, 1998, 102 (14), pp 2569–2577
    http://dx.doi.org/10.1021/jp972543+
'''


__all__ = ['get_ff', 'get_system']


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


def get_system(nmol, rho0=420.0*kilogram/meter**3):
    # Initial density
    masses = np.ones((nmol,))*( periodic[6].mass + 4*periodic[1].mass )
    # Initial volume
    V0 = np.sum(masses)/rho0
    # Estimate intermolecular distance based on the volume per molecule
    l0 = (V0/nmol)**(1.0/3.0)
    # Setup the system
    system = get_system_fluidum_grid(nmol, ffatypes=['CH4'], l0=l0)
    system.masses = masses
    return system
